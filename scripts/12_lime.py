"""Bước 5b: giải thích cục bộ bằng LIME (hộp đen) rồi đối chiếu với TreeSHAP gộp.

Quy trình:
  1. Tái tạo đúng mẫu luồng bị gắn cờ mà bước 5 (08_explain.py) đã giải thích, lấy tập con phân tầng cho LIME.
  2. Chạy LIME trên hàm dự đoán của cả hệ stacking (nền lấy từ tập train).
  3. Tính SHAP gộp cho cùng các luồng, so sánh: đặc trưng quan trọng, mức đồng thuận, ổn định, trung thực cục bộ.
Đầu ra: reports/lime_<ds>.json, data/processed/<ds>/lime_sample.parquet (hình vẽ ở bước report)
"""
import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from xsids.config import load_config, resolve_path
from xsids.explain.lime_stack import (
    build_explainer,
    deletion_test,
    global_importance,
    lime_weights,
    make_lime_predict_fn,
    make_proba_fn,
    rank_corr_rows,
    recover_sample_idx,
    stability_metrics,
    stratified_pick,
    summarize,
    topk_jaccard,
    topk_sign_agreement,
)
from xsids.explain.shap_stack import agreement_score, explain_stack, group_attribution
from xsids.features.engineer import GROUP_ORDER
from xsids.models.data import load_feature_set, load_split, xyw
from xsids.models.stacking import raw_matrix, stack_logit
from xsids.models.zoo import BASE_NAMES, load_models_cfg
from xsids.schema import ATTACK_COL, FAMILY_COL, LABEL_COL
from xsids.report.summaries import lime_text
from xsids.utils import get_logger, write_json

log = get_logger()
ap = argparse.ArgumentParser()
ap.add_argument("--dataset", choices=["unsw", "cse"], required=True)
ap.add_argument("--n", type=int, default=None, help="số luồng bị gắn cờ giải thích bằng LIME")
ap.add_argument("--num-perturbations", type=int, default=None, help="số điểm nhiễu mỗi luồng")
ap.add_argument("--background", type=int, default=None, help="số vector train làm nền")
ap.add_argument("--target", choices=["proba", "logit"], default=None)
ap.add_argument("--discretizer", choices=["quartile", "decile", "none"], default=None,
                help="none = không rời rạc hóa (nhiễu Gaussian quanh luồng)")
ap.add_argument("--kernel-width", type=float, default=None, help="độ rộng kernel của LIME (mặc định: 0,75·√d)")
ap.add_argument("--n-stab", type=int, default=None, help="số luồng dùng cho kiểm tra ổn định")
ap.add_argument("--stab-repeats", type=int, default=None, help="số lần lặp LIME (seed khác nhau) khi đo ổn định")
ap.add_argument("--shap-n", type=int, default=None, help="n của bước 08 (để tái tạo đúng mẫu SHAP)")
ap.add_argument("--fast", action="store_true", help="tham số nhỏ để chạy thử")
ap.add_argument("--processed-dir", default=None)
ap.add_argument("--models-dir", default=None)
ap.add_argument("--tag", default="")
args = ap.parse_args()

cfg, mcfg = load_config(), load_models_cfg()
lc = dict(mcfg["lime"])
lc.update((mcfg.get("lime_per_dataset") or {}).get(args.dataset) or {})  # cấu hình riêng của dataset (nếu có)
if args.fast:
    fl = mcfg["fast_lime"]
    lc.update({k: v for k, v in fl.items() if k != "stability"})
    lc["stability"] = {**lc["stability"], **fl.get("stability", {})}
n_lime = args.n or lc["n_samples"]
n_pert = args.num_perturbations or lc["num_perturbations"]
n_bg = args.background or lc["background_size"]
target = args.target or lc["target"]
disc = args.discretizer or lc["discretizer"]
kw = args.kernel_width if args.kernel_width is not None else lc["kernel_width"]
n_stab = args.n_stab or lc["stability"]["n_flows"]
reps = args.stab_repeats or lc["stability"]["repeats"]
K, ks = lc["top_k"], lc["deletion_ks"]
seed = mcfg["seed"]

proc = Path(args.processed_dir) if args.processed_dir else resolve_path(cfg, "processed_dir")
mdir = Path(args.models_dir) if args.models_dir else Path(cfg["_root"]) / "models" / args.dataset
rdir = resolve_path(cfg, "reports_dir")
fs = load_feature_set(proc)
feats, groups = fs["features"], fs["groups"]
models = {n: joblib.load(mdir / f"{n}.joblib") for n in BASE_NAMES}
st = joblib.load(mdir / "stack.joblib")
w, b, thr = st["w"], st["b"], st["threshold"]

# ── 1. mẫu luồng: tái tạo mẫu SHAP của bước 08, rồi lấy tập con cho LIME ─────
te = load_split(proc, args.dataset, "test")
X, y, wt, att = xyw(te, feats)
z_all = stack_logit(raw_matrix(models, X), w, b)
p_all = 1 / (1 + np.exp(-z_all))
flagged = np.flatnonzero(p_all >= thr)
log.info("%d/%d vector test bị gắn cờ (ngưỡng %.4f)", len(flagged), len(te), thr)

shap_n = args.shap_n or mcfg["explain"]["n_samples"]
sample_path = proc / args.dataset / "explain_sample.parquet"
sample_match = None
pick_shap = None
if sample_path.exists():
    ss = pd.read_parquet(sample_path)
    if "test_idx" in ss.columns:  # bước 08 mới: đọc thẳng chỉ số luồng đã giải thích
        pick_shap = ss["test_idx"].to_numpy(dtype=int)
        how = "đọc test_idx từ explain_sample.parquet"
    else:                          # file cũ: ghép lại theo (tên tấn công, logit z)
        pick_shap = recover_sample_idx(z_all, att, ss["z"].to_numpy(), ss[ATTACK_COL].astype(str).to_numpy())
        how = "ghép lại theo (tên tấn công, z) vì file cũ chưa có test_idx"
    if pick_shap is not None:
        pick_shap = np.sort(pick_shap)
        sample_match = bool(len(pick_shap) == len(ss) and np.allclose(np.sort(ss["z"].to_numpy()), np.sort(z_all[pick_shap]), atol=1e-6))
        log.info("Mẫu SHAP: %s (%d luồng) -> %s", how, len(pick_shap), "KHỚP" if sample_match else "KHÔNG khớp")
    else:
        log.warning("Không ghép lại được mẫu SHAP từ %s; hãy chạy lại bước 08 (cùng models) rồi chạy lại bước này", sample_path.name)
else:
    log.warning("Chưa có %s (chạy bước 08 trước nếu muốn so sánh với đúng mẫu SHAP đã lưu)", sample_path.name)
if pick_shap is None:  # dự phòng: tái tạo bằng cách lấy mẫu lại (có thể lệch nếu ngưỡng gắn cờ lệch giữa các lần chạy)
    pick_shap = stratified_pick(flagged, att, shap_n, np.random.default_rng(seed))

pick = stratified_pick(pick_shap, att, min(n_lime, len(pick_shap)), np.random.default_rng(seed + 1))
Xs = X.iloc[pick]
A_rows = Xs.to_numpy(dtype="float64")
log.info("LIME giải thích %d luồng (target=%s, %d điểm nhiễu, rời rạc hóa %s)", len(pick), target, n_pert, disc)

# ── 2. nền từ tập train, hàm dự đoán, chạy LIME ──────────────────────────────
tr = load_split(proc, args.dataset, "train")
Xtr, ytr, _, _ = xyw(tr, feats)
bg_idx = np.random.default_rng(seed + 2).choice(len(Xtr), size=min(n_bg, len(Xtr)), replace=False)
Xbg, ybg = Xtr.iloc[bg_idx].reset_index(drop=True), ytr[bg_idx]
benign = Xbg[ybg == 0]
baseline = (benign if len(benign) else Xbg).median().to_numpy(dtype="float64")  # giá trị "điển hình lành tính"
del tr, Xtr

predict_fn = make_lime_predict_fn(models, w, b, feats, target)
score_fn = make_proba_fn(models, w, b, feats)
t0 = time.time()


def progress(i, n):
    if i == 1 or i % max(1, n // 10) == 0 or i == n:
        log.info("  LIME %d/%d (%.1fs, ~%.2fs/luồng)", i, n, time.time() - t0, (time.time() - t0) / i)


explainer = build_explainer(Xbg, target, disc, kw, seed)
res = lime_weights(explainer, predict_fn, A_rows, target, n_pert, progress)
W = res["weights"]

# ── 3. SHAP gộp cho đúng các luồng này ───────────────────────────────────────
ex = explain_stack(models, w, b, Xs)
phi = ex["phi_stack"]
G_shap, gnames = group_attribution(phi, feats, groups, GROUP_ORDER)
G_lime, _ = group_attribution(W, feats, groups, GROUP_ORDER)
G_base = [group_attribution(ph, feats, groups, GROUP_ORDER)[0] for ph in ex["phi_per_base"]]
A_base = agreement_score(G_base)  # đồng thuận giữa 4 mô hình thành phần (bước 7)

# ── 4. độ đo ─────────────────────────────────────────────────────────────────
model_out = ex["z"] if target == "logit" else 1 / (1 + np.exp(-ex["z"]))
rho_feat = rank_corr_rows(np.abs(phi), np.abs(W))
jac = topk_jaccard(phi, W, K)
sgn = topk_sign_agreement(phi, W, K)
agr_group = agreement_score([G_shap, G_lime])
top1 = float((np.argmax(np.abs(phi), axis=1) == np.argmax(np.abs(W), axis=1)).mean())

imp_shap, imp_lime = global_importance(phi), global_importance(W)
g_rho = float(rank_corr_rows(imp_shap[None, :], imp_lime[None, :])[0])
g_top = set(np.argsort(-imp_shap)[:K]) & set(np.argsort(-imp_lime)[:K])

fam_sub = te.iloc[pick][FAMILY_COL].astype(str).to_numpy()
by_family = {}
for f in np.unique(fam_sub):
    m = fam_sub == f
    s_top = np.argsort(-np.abs(phi[m]).mean(axis=0))[:K]
    l_top = np.argsort(-np.abs(W[m]).mean(axis=0))[:K]
    by_family[f] = {"n": int(m.sum()), "shap_top": [feats[i] for i in s_top], "lime_top": [feats[i] for i in l_top],
                    "overlap": len(set(s_top) & set(l_top)), "k": K}

log.info("Kiểm tra trung thực (xóa đặc trưng)...")
dele = deletion_test(score_fn, A_rows, {"shap": phi, "lime": W}, baseline, ks, thr, np.random.default_rng(seed + 3))
dele_naive = deletion_test(score_fn, A_rows, {"shap": phi, "lime": W}, baseline, ks, thr,
                           np.random.default_rng(seed + 3), eligible_only=False)

log.info("Kiểm tra ổn định: %d luồng × %d lần chạy...", min(n_stab, len(pick)), reps)
sub = stratified_pick(np.arange(len(pick)), att[pick], min(n_stab, len(pick)), np.random.default_rng(seed + 4))
runs = []
for r in range(reps):
    ex_r = build_explainer(Xbg, target, disc, kw, seed + 100 + r)
    runs.append(lime_weights(ex_r, predict_fn, A_rows[sub], target, n_pert)["weights"])
    log.info("  lần %d/%d xong", r + 1, reps)
stab = stability_metrics(np.stack(runs), K)
stab["shap"] = "xác định (không ngẫu nhiên): spearman = jaccard = 1.0 theo cấu trúc"

ok = ~np.isnan(agr_group) & ~np.isnan(A_base)
a_vs = float(rank_corr_rows(agr_group[ok][None, :], A_base[ok][None, :])[0]) if ok.sum() > 2 else None

summary = {
    "dataset": args.dataset, "target": target, "n_lime": int(len(pick)), "num_perturbations": n_pert,
    "background_size": int(len(Xbg)), "discretizer": disc, "kernel_width": kw, "top_k": K,
    "sample_matches_explain_sample": sample_match,
    "local_fidelity": {
        "lime_r2": summarize(res["r2"]),
        "frac_r2_below_0.5": float((res["r2"] < 0.5).mean()),
        "local_pred_abs_error": summarize(np.abs(res["local_pred"] - model_out)),
    },
    "lime_top1_feature_counts": {feats[j]: int(c) for j, c in zip(*np.unique(np.argmax(np.abs(W), axis=1), return_counts=True))},
    "shap_top1_feature_counts": {feats[j]: int(c) for j, c in zip(*np.unique(np.argmax(np.abs(phi), axis=1), return_counts=True))},
    "agreement_shap_lime": {
        "feature_spearman_abs": summarize(rho_feat), "topk_jaccard": summarize(jac),
        "topk_sign_agreement": summarize(sgn), "group_spearman": summarize(agr_group), "top1_same_rate": top1,
    },
    "global_importance": {
        "features": feats, "shap": imp_shap, "lime": imp_lime, "spearman": g_rho, "topk_overlap": len(g_top),
    },
    "importance_by_family": by_family,
    "deletion_test": dele,
    "deletion_test_unfiltered": dele_naive,
    "stability_lime": stab,
    "agreement_vs_base_models": {"spearman(shap-lime group agreement, A(x))": a_vs},
    "additivity_ok_shap": bool(all(v.max() <= mcfg["explain"]["tolerance"] for v in ex["additivity_error"].values())),
    "seconds": round(time.time() - t0, 1),
}
stem = f"lime_{args.dataset}" + (f"_{args.tag}" if args.tag else "")
write_json(summary, rdir / f"{stem}.json")

# ── 5. bảng từng luồng ───────────────────────────────────────────────────────
out = pd.DataFrame({"test_idx": pick, ATTACK_COL: att[pick], FAMILY_COL: fam_sub, LABEL_COL: y[pick],
                    "p": p_all[pick], "z": z_all[pick], "lime_r2": res["r2"], "lime_local_pred": res["local_pred"],
                    "lime_intercept": res["intercept"], "spearman_feat": rho_feat, f"jaccard_top{K}": jac,
                    f"sign_agree_top{K}": sgn, "agreement_group": agr_group, "agreement_base": A_base})
for j, f in enumerate(feats):
    out[f"lime_{f}"], out[f"shap_{f}"] = W[:, j], phi[:, j]
for j, g in enumerate(gnames):
    out[f"lime_g_{g}"], out[f"shap_g_{g}"] = G_lime[:, j], G_shap[:, j]
sample_out = proc / args.dataset / (f"lime_sample_{args.tag}.parquet" if args.tag else "lime_sample.parquet")
out.to_parquet(sample_out, index=False)

# ── 6. tóm tắt (cùng định dạng với scripts/13_summary.py) ───────────────────
print(lime_text(args.dataset, summary))
log.info("Đã ghi reports/%s.json và %s", stem, sample_out)
