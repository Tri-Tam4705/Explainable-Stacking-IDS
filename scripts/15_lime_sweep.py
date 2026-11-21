"""Chọn cấu hình LIME: thử nhanh nhiều cấu hình trên một tập nhỏ luồng rồi in bảng so sánh.

Vì sao cần: ở lần chạy đầy đủ, LIME cho R² mô hình thay thế rất thấp (UNSW 0,24; CSE 0,49) và hạng 1 luôn là một đặc trưng
(server_psh 271/271 trên UNSW, duration_in_ms 233/233 trên CSE). Script này thử các cách cấu hình (đại lượng giải thích,
rời rạc hóa, độ rộng kernel, số điểm nhiễu) trên cùng một tập luồng nhỏ, đo:
    R² thay thế        độ khớp của mô hình tuyến tính cục bộ (không dùng SHAP, nên không thiên vị)
    top1 tập trung     tỷ lệ luồng có chung đặc trưng hạng 1 (1,0 là đáng ngờ)
    ổn định            Spearman giữa hai lần chạy khác seed
    trung thực xóa     (LIME − ngẫu nhiên)/(SHAP − ngẫu nhiên) khi xóa k đặc trưng; phép thử này có thiên vị theo SHAP
Cấu hình đề xuất được chọn theo R² (có điều kiện ổn định), vì R² không phụ thuộc SHAP. Chọn theo phép xóa sẽ làm
LIME "học" theo thước đo dùng để đánh giá nó.

Dùng:
    python scripts/15_lime_sweep.py --dataset unsw
    python scripts/15_lime_sweep.py --dataset cse --n 40 --fast
Mất khoảng vài phút mỗi dataset (40 luồng × ~10 cấu hình × ~0,3 s). Không ghi đè kết quả LIME cũ.
Đầu ra: reports/lime_sweep_<ds>.json và dòng lệnh gợi ý để chạy lại bước 12.
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
    topk_jaccard,
)
from xsids.explain.shap_stack import explain_stack
from xsids.models.data import load_feature_set, load_split, xyw
from xsids.models.stacking import raw_matrix, stack_logit
from xsids.models.zoo import BASE_NAMES, load_models_cfg
from xsids.schema import ATTACK_COL
from xsids.utils import get_logger, write_json

log = get_logger()
ap = argparse.ArgumentParser()
ap.add_argument("--dataset", choices=["unsw", "cse"], required=True)
ap.add_argument("--n", type=int, default=40, help="số luồng dùng để thử mỗi cấu hình")
ap.add_argument("--background", type=int, default=5000, help="số vector train làm nền")
ap.add_argument("--stab-flows", type=int, default=15, help="số luồng dùng đo ổn định (2 lần chạy)")
ap.add_argument("--fast", action="store_true", help="ít điểm nhiễu hơn, chạy thử")
ap.add_argument("--processed-dir", default=None)
ap.add_argument("--models-dir", default=None)
args = ap.parse_args()

cfg, mcfg = load_config(), load_models_cfg()
lc = mcfg["lime"]
seed, K, ks = mcfg["seed"], lc["top_k"], lc["deletion_ks"]
proc = Path(args.processed_dir) if args.processed_dir else resolve_path(cfg, "processed_dir")
mdir = Path(args.models_dir) if args.models_dir else Path(cfg["_root"]) / "models" / args.dataset
rdir = resolve_path(cfg, "reports_dir")
fs = load_feature_set(proc)
feats = fs["features"]
d = len(feats)
models = {n: joblib.load(mdir / f"{n}.joblib") for n in BASE_NAMES}
st = joblib.load(mdir / "stack.joblib")
w, b, thr = st["w"], st["b"], st["threshold"]

# ── tập luồng: con của mẫu SHAP đã lưu (khớp với bước 08/12) ─────────────────
te = load_split(proc, args.dataset, "test")
X, y, wt, att = xyw(te, feats)
z_all = stack_logit(raw_matrix(models, X), w, b)
flagged = np.flatnonzero(1 / (1 + np.exp(-z_all)) >= thr)
sp = proc / args.dataset / "explain_sample.parquet"
pool = None
if sp.exists():
    ss = pd.read_parquet(sp)
    pool = ss["test_idx"].to_numpy(dtype=int) if "test_idx" in ss.columns else \
        recover_sample_idx(z_all, att, ss["z"].to_numpy(), ss[ATTACK_COL].astype(str).to_numpy())
if pool is None:
    log.warning("Không lấy được mẫu SHAP đã lưu; dùng luồng bị gắn cờ ngẫu nhiên phân tầng")
    pool = flagged
pick = stratified_pick(np.sort(pool), att, min(args.n, len(pool)), np.random.default_rng(seed + 7))
Xs = X.iloc[pick]
A_rows = Xs.to_numpy(dtype="float64")
log.info("%s: thử cấu hình LIME trên %d luồng bị gắn cờ (%d tên tấn công)", args.dataset, len(pick), len(np.unique(att[pick])))

tr = load_split(proc, args.dataset, "train")
Xtr, ytr, _, _ = xyw(tr, feats)
bg_idx = np.random.default_rng(seed + 2).choice(len(Xtr), size=min(args.background, len(Xtr)), replace=False)
Xbg, ybg = Xtr.iloc[bg_idx].reset_index(drop=True), ytr[bg_idx]
benign = Xbg[ybg == 0]
baseline = (benign if len(benign) else Xbg).median().to_numpy(dtype="float64")
del tr, Xtr

score_fn = make_proba_fn(models, w, b, feats)
phi = explain_stack(models, w, b, Xs)["phi_stack"]
imp_shap = global_importance(phi)

n_pert_full = 1000 if args.fast else 5000
sqrt_d = float(np.sqrt(d))
# (tên, target, discretizer, hệ số kernel_width/√d hoặc None = mặc định 0,75, số điểm nhiễu)
GRID = [
    ("proba/quartile (LIME gốc)", "proba", "quartile", None, n_pert_full),
    ("proba/decile", "proba", "decile", None, n_pert_full),
    ("logit/quartile", "logit", "quartile", None, n_pert_full),
    ("logit/decile", "logit", "decile", None, n_pert_full),
    ("logit/none", "logit", "none", None, n_pert_full),
    ("logit/none kernel×2", "logit", "none", 1.5, n_pert_full),
    ("logit/none kernel×4", "logit", "none", 3.0, n_pert_full),
    ("proba/none", "proba", "none", None, n_pert_full),
    ("logit/decile 2× nhiễu", "logit", "decile", None, 2 * n_pert_full),
    ("logit/none 2× nhiễu", "logit", "none", None, 2 * n_pert_full),
]

rows = []
for name, target, disc, kmul, n_pert in GRID:
    t0 = time.time()
    kw = None if kmul is None else kmul * sqrt_d
    pf = make_lime_predict_fn(models, w, b, feats, target)
    res = lime_weights(build_explainer(Xbg, target, disc, kw, seed), pf, A_rows, target, n_pert)
    W = res["weights"]
    dele = deletion_test(score_fn, A_rows, {"shap": phi, "lime": W}, baseline, ks, thr, np.random.default_rng(seed + 3))
    ratios = {}
    for k in (3, 5):
        dk = dele[str(k)]
        den = dk["shap"]["mean_drop"] - dk["random"]["mean_drop"]
        ratios[k] = float((dk["lime"]["mean_drop"] - dk["random"]["mean_drop"]) / den) if den > 1e-9 else float("nan")
    top1 = np.argmax(np.abs(W), axis=1)
    vals, cnt = np.unique(top1, return_counts=True)
    j = int(np.argmax(cnt))
    # ổn định: 2 lần chạy khác seed trên tập con
    sub = np.arange(min(args.stab_flows, len(A_rows)))
    runs = [lime_weights(build_explainer(Xbg, target, disc, kw, seed + 100 + r), pf, A_rows[sub], target, n_pert)["weights"]
            for r in range(2)]
    stab = stability_metrics(np.stack(runs), K)
    imp_l = global_importance(W)
    row = {
        "config": name, "target": target, "discretizer": disc, "kernel_mult": kmul, "num_perturbations": n_pert,
        "r2_mean": float(res["r2"].mean()), "frac_r2_lt_0.5": float((res["r2"] < 0.5).mean()),
        "top1_feature": feats[int(vals[j])], "top1_share": float(cnt[j] / len(top1)),
        "stab_spearman": stab["spearman"]["mean"], "stab_jaccard": stab["topk_jaccard"]["mean"],
        "deletion_ratio_k3": ratios[3], "deletion_ratio_k5": ratios[5],
        "global_spearman_vs_shap": float(rank_corr_rows(imp_shap[None, :], imp_l[None, :])[0]),
        "jaccard_top5_vs_shap": float(topk_jaccard(phi, W, K).mean()),
        "seconds": round(time.time() - t0, 1),
    }
    rows.append(row)
    log.info("%-26s R²=%.3f top1=%s(%.0f%%) stab=%.2f xóa k3/k5=%.0f%%/%.0f%% (%.0fs)", name, row["r2_mean"], row["top1_feature"],
             100 * row["top1_share"], row["stab_spearman"], 100 * ratios[3], 100 * ratios[5], row["seconds"])

# ── chọn: R² cao nhất trong các cấu hình ổn định (Spearman ≥ 0,7), không dùng SHAP ─
ok = [r for r in rows if r["stab_spearman"] is not None and r["stab_spearman"] >= 0.7] or rows
best = max(ok, key=lambda r: r["r2_mean"])
flags = f"--target {best['target']} --discretizer {best['discretizer']}"
if best["kernel_mult"] is not None:
    flags += f" --kernel-width {best['kernel_mult'] * sqrt_d:.3f}"
flags += f" --num-perturbations {best['num_perturbations']}"
write_json({"dataset": args.dataset, "n_flows": int(len(pick)), "rows": rows, "recommended": best, "flags": flags},
           rdir / f"lime_sweep_{args.dataset}.json")

hdr = f"{'cấu hình':26s} {'R²':>6s} {'R²<.5':>6s} {'top1 (tỷ lệ)':>28s} {'ổn định':>8s} {'xóa k3':>7s} {'xóa k5':>7s} {'ρ toàn cục':>10s}"
print(f"\nSO SÁNH CẤU HÌNH LIME — {args.dataset.upper()} ({len(pick)} luồng)\n" + "─" * len(hdr) + "\n" + hdr)
for r in rows:
    mark = " ◄" if r is best else ""
    print(f"{r['config']:26s} {r['r2_mean']:6.3f} {100 * r['frac_r2_lt_0.5']:5.0f}% "
          f"{r['top1_feature'] + ' (' + format(100 * r['top1_share'], '.0f') + '%)':>28s} {r['stab_spearman']:8.2f} "
          f"{100 * r['deletion_ratio_k3']:6.0f}% {100 * r['deletion_ratio_k5']:6.0f}% {r['global_spearman_vs_shap']:10.2f}{mark}")
print("\nR²: độ khớp mô hình thay thế · top1: đặc trưng hạng 1 phổ biến nhất · xóa kN: (LIME−ngẫu nhiên)/(SHAP−ngẫu nhiên)")
print("Đề xuất (R² cao nhất trong các cấu hình ổn định, không dùng SHAP để chọn):", best["config"])
print("\nChạy thử cấu hình đề xuất mà không ghi đè kết quả chính (ghi ra reports/lime_<ds>_thu.json):")
print(f"  python scripts/12_lime.py --dataset {args.dataset} {flags} --tag thu")
print("Muốn dùng làm cấu hình chính: sửa mục lime / lime_per_dataset trong configs/models.yaml rồi chạy python main.py lime.")
log.info("Đã ghi reports/lime_sweep_%s.json", args.dataset)
