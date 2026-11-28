"""Kiểm toán nhãn: nhãn kế thừa từ bộ gốc có đáng tin không, và nhãn nghi sai ảnh hưởng kết quả đến mức nào.

Không có gói tin gốc nên không gán lại nhãn từ nguồn được. Script đo ba mặt kiểm tra được trên chính dữ liệu:

  1. Trùng vector khác nhãn. Nhiều luồng có cùng 25 đặc trưng nhưng nhãn khác nhau (Benign và tấn công). Với mỗi tên tấn
     công: tỷ lệ luồng nằm ở vector có nhãn Benign chiếm đa số. Không mô hình nào chỉ dùng 25 đặc trưng bắt được phần này,
     nên nó là mức trần của recall (xsids.data.overlap).
  2. Học tự tin (confident learning, Northcutt và cộng sự, 2021). Xác suất out-of-fold của LightGBM trên mọi vector duy nhất
     (gộp train, val, test; 5 fold phân tầng theo tên tấn công). Vector mang nhãn tấn công bị nghi sai nhãn khi xác suất
     Benign của nó không nhỏ hơn ngưỡng tự tin của lớp Benign (trung bình xác suất Benign trên các vector Benign); chiều
     ngược lại tương tự. Kết quả là ước lượng, không phải nhãn đúng.
  3. Ảnh hưởng. Chỉ số của mô hình chính trên test khi bỏ các vector nghi sai nhãn (cận trên, nếu toàn bộ chúng sai thật),
     và (--retrain) chỉ số khi huấn luyện lại cả hệ trên train và val đã bỏ vector nghi sai.

Dùng:
    python scripts/27_label_audit.py --dataset cse --retrain
Đầu ra: reports/label_audit_<ds>.json. Không ghi đè mô hình hay kết quả chính.
"""
import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit

import _bootstrap  # noqa: F401
from xsids.config import load_config, resolve_path
from xsids.data.overlap import vector_hash
from xsids.data.split import make_strata
from xsids.models.data import load_feature_set, load_split, natural_prevalence, xyw
from xsids.models.evaluate import evaluate
from xsids.models.stacking import raw_matrix, stack_logit
from xsids.models.zoo import BASE_NAMES, get_params, load_models_cfg, make_model, raw_output
from xsids.utils import get_logger, write_json

log = get_logger()
ap = argparse.ArgumentParser()
ap.add_argument("--dataset", choices=["unsw", "cse"], required=True)
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--retrain", action="store_true")
ap.add_argument("--fast", action="store_true")
args = ap.parse_args()

ds = args.dataset
cfg, mcfg = load_config(), load_models_cfg()
proc = resolve_path(cfg, "processed_dir")
feats = load_feature_set(proc)["features"]
seed, n_jobs = mcfg["seed"], mcfg["n_jobs"]
params = get_params(mcfg, args.fast)
pi = natural_prevalence(proc, ds)
t_start = time.time()
KEYS = ("dr", "far", "macro_f1", "pr_auc", "precision_at_natural_prevalence")

parts = {s: load_split(proc, ds, s) for s in ("train", "val", "test")}
pool = pd.concat([parts[s].assign(_split=s) for s in parts], ignore_index=True)
pool["attack"] = pool["attack"].astype(str)
X, y, w, a = xyw(pool, feats)

# 1) trùng vector khác nhãn, theo 25 đặc trưng
h = vector_hash(pool, feats)
t = pd.DataFrame({"h": h, "y": y, "w": w})
B = t[t.y == 0].groupby("h")["w"].sum()
A = t[t.y == 1].groupby("h")["w"].sum()
Bh = B.reindex(h, fill_value=0).to_numpy()
Ah = A.reindex(h, fill_value=0).to_numpy()
benign_major = Bh > Ah          # hòa thì coi là tấn công, như quy tắc gộp trùng
overlap = {}
for c in sorted(set(a)):
    m = a == c
    lab = int(y[m][0])
    shared = (Bh[m] > 0) & (Ah[m] > 0)
    lost = benign_major[m] if lab == 1 else ~benign_major[m]
    overlap[c] = {"label": lab, "n_vectors": int(m.sum()), "n_weighted": int(w[m].sum()),
                  "shared_with_other_label": float(w[m][shared].sum() / w[m].sum()),
                  "unreachable": float(w[m][lost].sum() / w[m].sum())}
att_w = w[y == 1]
overlap_all = {"attack_recall_ceiling": float(att_w[~benign_major[y == 1]].sum() / att_w.sum()),
               "benign_far_floor": float(w[(y == 0) & ~benign_major].sum() / w[y == 0].sum())}
log.info("Trần recall do trùng vector khác nhãn: %.4f", overlap_all["attack_recall_ceiling"])

# 2) học tự tin với xác suất out-of-fold
from sklearn.model_selection import StratifiedKFold  # noqa: E402

p = np.zeros(len(pool))
skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=seed)
for k, (tr, te) in enumerate(skf.split(np.zeros(len(pool)), make_strata(pd.Series(a), args.folds * 2))):
    t0 = time.time()
    m = make_model("lgbm", params["lgbm"], seed, n_jobs).fit(X.iloc[tr], y[tr])
    p[te] = expit(raw_output("lgbm", m, X.iloc[te]))
    log.info("OOF fold %d: %.0f giây", k + 1, time.time() - t0)
t_benign = float((1 - p[y == 0]).mean())
t_attack = float(p[y == 1].mean())
suspect = np.where(y == 1, (1 - p) >= t_benign, p >= t_attack)
cl = {"threshold_benign": t_benign, "threshold_attack": t_attack,
      "suspect_share_vectors": float(suspect.mean()), "suspect_share_weighted": float(w[suspect].sum() / w.sum()),
      "by_attack": {}}
for c in sorted(set(a)):
    m = a == c
    cl["by_attack"][c] = {"n_vectors": int(m.sum()), "n_weighted": int(w[m].sum()),
                          "suspect_vectors": int(suspect[m].sum()),
                          "suspect_share_weighted": float(w[m & suspect].sum() / w[m].sum()),
                          "suspect_also_unreachable": float(w[m & suspect & (benign_major if y[m][0] == 1 else ~benign_major)].sum()
                                                            / max(w[m & suspect].sum(), 1e-12))}
log.info("Nghi sai nhãn: %.2f%% vector, %.2f%% luồng", 100 * cl["suspect_share_vectors"], 100 * cl["suspect_share_weighted"])

# 3) ảnh hưởng lên mô hình chính
mdir = Path(cfg["_root"]) / "models" / ds
st = joblib.load(mdir / "stack.joblib")
models = {n: joblib.load(mdir / f"{n}.joblib") for n in BASE_NAMES}
is_te = pool["_split"].to_numpy() == "test"
Xte, yte, wte, ate = X[is_te], y[is_te], w[is_te], a[is_te]
s_te = expit(stack_logit(raw_matrix(models, Xte), st["w"], st["b"]))
clean_te = ~suspect[is_te]


def ev(s, thr, mask):
    r = evaluate(yte[mask], s[mask], wte[mask], thr, ate[mask], pi)
    return {"weighted": {k: r["weighted"][k] for k in KEYS},
            "per_attack_recall": {k: v["recall_weighted"] for k, v in r["per_attack_recall"].items()}}


effect = {"main_model": {"test_all": ev(s_te, st["threshold"], np.ones(len(yte), bool)),
                         "test_without_suspects": ev(s_te, st["threshold"], clean_te)},
          "n_test_suspects": int((~clean_te).sum()), "n_test_suspects_weighted": int(wte[~clean_te].sum())}

if args.retrain:
    from xsids.models.experiment import fit_system, scores as sys_scores
    from xsids.models.evaluate import best_threshold

    t0 = time.time()
    keep_tr = ~suspect[pool["_split"].to_numpy() == "train"]
    keep_va = ~suspect[pool["_split"].to_numpy() == "val"]
    tr = parts["train"][keep_tr].reset_index(drop=True)
    va = parts["val"][keep_va].reset_index(drop=True)
    Xtr, ytr, _, atr = xyw(tr, feats)
    Xva, yva, wva, _ = xyw(va, feats)
    system = fit_system(Xtr, ytr, atr, params, seed, n_jobs, mcfg["cv_folds"], mcfg["meta"]["C"])
    s_va = sys_scores(system, Xva)["stack_explainable"]
    thr = best_threshold(yva, s_va, wva)
    s_new = sys_scores(system, Xte)["stack_explainable"]
    effect["retrained_without_suspects"] = {"n_train": int(len(tr)), "threshold": float(thr),
                                            "test_all": ev(s_new, thr, np.ones(len(yte), bool)),
                                            "test_without_suspects": ev(s_new, thr, clean_te),
                                            "seconds": round(time.time() - t0, 1)}
    log.info("Huấn luyện lại không có vector nghi sai: %.0f giây", time.time() - t0)

out = {"dataset": ds, "fast": args.fast, "n_vectors": int(len(pool)), "n_weighted": int(w.sum()),
       "overlap": {"overall": overlap_all, "by_attack": overlap}, "confident_learning": cl, "effect": effect,
       "seconds": round(time.time() - t_start, 1)}
write_json(out, resolve_path(cfg, "reports_dir") / f"label_audit_{ds}.json")
x0 = effect["main_model"]["test_all"]["weighted"]
x1 = effect["main_model"]["test_without_suspects"]["weighted"]
log.info("%s: mô hình chính DR %.4f -> %.4f, FAR %.6f -> %.6f khi bỏ vector test nghi sai nhãn",
         ds, x0["dr"], x1["dr"], x0["far"], x1["far"])
