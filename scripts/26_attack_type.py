"""Gọi tên loại tấn công cho luồng bị gắn cờ: bước hai đa lớp, khắc phục giới hạn chỉ phân loại nhị phân.

Hệ chính trả lời "tấn công hay không". Bước này huấn luyện thêm một mô hình đa lớp LightGBM chỉ trên các vector tấn công
của train, với nhãn là tên tấn công, để gọi tên loại cho luồng bị gắn cờ. Hệ stacking nhị phân và lời giải thích của nó
giữ nguyên; bước hai chỉ thêm một đầu ra.

  1. Chọn cấu hình (tốc độ học, số lá, trọng số lớp) theo macro-F1 trên vector tấn công của val. Tham số của mô hình
     nhị phân (tốc độ học 0,1, 400 cây) làm LightGBM đa lớp phân kỳ (độ chính xác trên chính train chỉ khoảng 0,5 với
     UNSW), nên bước này dùng tốc độ học 0,05.
  2. Trên vector tấn công của test: precision, recall, F1 từng loại (có trọng số n_orig), macro-F1 và độ chính xác.
  3. Mức trần: cùng một vector 25 đặc trưng có thể mang nhiều tên tấn công; tỷ lệ luồng rơi vào vector mà tên của nó
     không chiếm đa số là phần không mô hình nào gọi đúng được (xem xsids.data.overlap.overlap_ceiling).
  4. Đầu cuối: tỷ lệ luồng tấn công vừa được hệ nhị phân chính gắn cờ vừa được gọi đúng tên, trên phần test mà cả hai
     mô hình chưa từng thấy.
  5. Giải thích: TreeSHAP đa lớp của LightGBM (pred_contrib), nhóm đặc trưng có |φ| trung bình lớn nhất cho từng loại.

Dùng:
    python scripts/26_attack_type.py --dataset unsw
    python scripts/26_attack_type.py --dataset cse
Đầu ra: reports/attack_type_<ds>.json.
"""
import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import precision_recall_fscore_support

import _bootstrap  # noqa: F401
from xsids.config import load_config, resolve_path
from xsids.data.overlap import overlap_ceiling, vector_hash
from xsids.features.engineer import GROUP_ORDER
from xsids.models.data import load_feature_set, load_split, xyw
from xsids.models.stacking import raw_matrix, stack_logit
from xsids.models.zoo import BASE_NAMES, load_models_cfg
from xsids.utils import get_logger, write_json

log = get_logger()
ap = argparse.ArgumentParser()
ap.add_argument("--dataset", choices=["unsw", "cse"], required=True)
ap.add_argument("--n-explain", type=int, default=500, help="số vector test mỗi loại dùng để tính TreeSHAP đa lớp")
ap.add_argument("--fast", action="store_true")
args = ap.parse_args()

ds = args.dataset
cfg, mcfg = load_config(), load_models_cfg()
root = Path(cfg["_root"])
proc = resolve_path(cfg, "processed_dir")
fs = load_feature_set(proc)
feats, groups = fs["features"], fs["groups"]
seed, n_jobs = mcfg["seed"], mcfg["n_jobs"]
GRID = [{"learning_rate": 0.05, "n_estimators": n, "num_leaves": l, "class_weight": cw}
        for n, l in ((300, 31), (300, 63)) for cw in (None, "balanced")]
if args.fast:
    GRID = [{"learning_rate": 0.1, "n_estimators": 60, "num_leaves": 15, "class_weight": None}]
t_start = time.time()

parts = {s: load_split(proc, ds, s) for s in ("train", "val", "test")}
att = {s: d[d["label"] == 1].reset_index(drop=True) for s, d in parts.items()}
for s in att:
    att[s]["attack"] = att[s]["attack"].astype(str)
classes = sorted(att["train"]["attack"].unique())
log.info("%s: %s vector tấn công train, %d loại", ds,
         len(att["train"]), len(classes))


def fit(cfg_):
    from lightgbm import LGBMClassifier
    X, _, _, a = xyw(att["train"], feats)
    return LGBMClassifier(objective="multiclass", subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1,
                          random_state=seed, n_jobs=n_jobs, **cfg_).fit(X, a)


def scores(clf, df):
    X, _, w, a = xyw(df, feats)
    pred = clf.predict(X)
    labels = sorted(set(classes) | set(a))
    p, r, f, _ = precision_recall_fscore_support(a, pred, labels=labels, sample_weight=w, zero_division=0)
    per = {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
               "n_vectors": int((a == c).sum()), "n_weighted": int(w[a == c].sum())} for i, c in enumerate(labels)}
    present = [c for c in labels if per[c]["n_vectors"] > 0]
    return {"macro_f1": float(np.mean([per[c]["f1"] for c in present])),
            "accuracy": float(w[pred == a].sum() / w.sum()), "per_class": per}, pred


# 1) chọn cấu hình trên val
cand = {}
for g in GRID:
    t = time.time()
    key = "lr{learning_rate}_n{n_estimators}_l{num_leaves}_{class_weight}".format(**g)
    clf = fit(g)
    cand[key] = (clf, scores(clf, att["val"])[0])
    log.info("%s: macro-F1 val %.4f, độ chính xác val %.4f (%.0f giây)", key, cand[key][1]["macro_f1"],
             cand[key][1]["accuracy"], time.time() - t)
best = max(cand, key=lambda k: cand[k][1]["macro_f1"])
clf = cand[best][0]
train_acc = scores(clf, att["train"])[0]["accuracy"]

# 2) test
res_test, pred_te = scores(clf, att["test"])
X_te, _, w_te, a_te = xyw(att["test"], feats)
conf = pd.crosstab(pd.Series(a_te, name="true"), pd.Series(pred_te, name="pred"), values=w_te, aggfunc="sum").fillna(0)
top_conf = {}
for c in conf.index:
    row = conf.loc[c] / conf.loc[c].sum()
    other = row.drop(c, errors="ignore").sort_values(ascending=False)
    top_conf[c] = {k: float(v) for k, v in other.head(2).items() if v > 0}

# 3) mức trần do vector trùng mang nhiều tên (gộp train + val + test, theo 25 đặc trưng)
pool = pd.concat([att[s] for s in ("train", "val", "test")], ignore_index=True)
ceil = overlap_ceiling(vector_hash(pool, feats), pool["attack"].to_numpy(), pool["n_orig"].to_numpy())

# 4) đầu cuối với hệ nhị phân chính
mdir = root / "models" / ds
st = joblib.load(mdir / "stack.joblib")
models = {n: joblib.load(mdir / f"{n}.joblib") for n in BASE_NAMES}
te = att["test"]
keep = np.ones(len(te), dtype=bool)
detected = expit(stack_logit(raw_matrix(models, X_te), st["w"], st["b"])) >= st["threshold"]
end = {}
for c in sorted(set(a_te)):
    m = (a_te == c) & keep
    if m.any():
        ww = w_te[m]
        end[c] = {"detected": float(ww[detected[m]].sum() / ww.sum()),
                  "detected_and_named": float(ww[detected[m] & (pred_te[m] == c)].sum() / ww.sum()),
                  "n_weighted": int(ww.sum())}
mk = keep
end_all = {"detected": float(w_te[mk & detected].sum() / w_te[mk].sum()),
           "detected_and_named": float(w_te[mk & detected & (pred_te == a_te)].sum() / w_te[mk].sum()),
           "named_given_detected": float(w_te[mk & detected & (pred_te == a_te)].sum() / max(w_te[mk & detected].sum(), 1e-12))}

# 5) TreeSHAP đa lớp: nhóm quan trọng nhất cho từng loại
rng = np.random.default_rng(0)
cls_order = list(clf.classes_)
d = len(feats)
gidx = {g: [i for i, f in enumerate(feats) if groups[f] == g] for g in GROUP_ORDER if any(groups[f] == g for f in feats)}
explain = {}
for c in cls_order:
    idx = np.flatnonzero(a_te == c)
    if len(idx) == 0:
        continue
    idx = np.sort(rng.choice(idx, size=min(args.n_explain, len(idx)), replace=False))
    contrib = clf.predict(X_te.iloc[idx], pred_contrib=True)
    k = cls_order.index(c)
    phi = np.asarray(contrib)[:, k * (d + 1):(k + 1) * (d + 1) - 1]
    imp = {g: float(np.abs(phi[:, ii].sum(axis=1)).mean()) for g, ii in gidx.items()}
    explain[c] = {"mean_abs_group": imp, "top_group": max(imp, key=imp.get), "n": int(len(idx))}

out = {"dataset": ds, "fast": args.fast, "config": best, "train_accuracy": train_acc,
       "val": {k: {"macro_f1": v[1]["macro_f1"], "accuracy": v[1]["accuracy"]} for k, v in cand.items()}, "n_train_attack_vectors": int(len(att["train"])),
       "test": res_test, "top_confusions": top_conf, "ceiling": ceil,
       "end_to_end": {"overall": end_all, "per_class": end, "n_test_vectors_used": int(keep.sum())},
       "explain": explain, "seconds": round(time.time() - t_start, 1)}
write_json(out, resolve_path(cfg, "reports_dir") / f"attack_type_{ds}.json")
log.info("%s: macro-F1 %.4f, độ chính xác %.4f (trần %.4f); đầu cuối gắn cờ và gọi đúng tên %.4f",
         ds, res_test["macro_f1"], res_test["accuracy"], ceil["accuracy_ceiling"], end_all["detected_and_named"])
