"""Bước 7: Ablation điểm đồng thuận A(x) — cắt lần lượt từng nhóm đặc trưng,
đo hiệu năng giảm bao nhiêu và điểm đồng thuận thay đổi như thế nào.

Ablation strategy: thay cột của nhóm bị cắt bằng một giá trị "trung lập" rồi chạy lại inference (không huấn luyện lại)
→ đo DR/FAR/macro-F1 mất đi bao nhiêu.
  --fill median (mặc định): median của tập validation. Giá trị này nằm trong phân phối thật nên ít nhiễu hơn.
  --fill zero              : đặt bằng 0 (cách cũ). 0 thường nằm ngoài phân phối nên ΔFAR bị thổi phồng; chỉ dùng để đối chứng.

Lưu ý về A(x): "ablation A(x)" ở cuối script KHÔNG chạy lại mô hình. Nó bỏ cột đóng góp SHAP của nhóm bị cắt khỏi các vector
nhóm đã lưu (explain_sample.parquet) rồi tính lại độ đồng thuận giữa 4 mô hình trên các nhóm còn lại. Vì vậy ΔA(x) đo độ nhạy
của chính công thức A(x) với số nhóm, không phải ảnh hưởng của việc bỏ đặc trưng lên mô hình.
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from xsids.config import load_config, resolve_path
from xsids.explain.shap_stack import agreement_score
from xsids.features.engineer import GROUP_ORDER
from xsids.models.data import load_feature_set, load_split, natural_prevalence, xyw
from xsids.models.evaluate import best_threshold, evaluate
from xsids.models.stacking import raw_matrix, stack_logit
from xsids.models.zoo import BASE_NAMES, load_models_cfg
from xsids.report.summaries import ablation_text
from xsids.utils import get_logger, write_json

log = get_logger()
ap = argparse.ArgumentParser()
ap.add_argument("--dataset", choices=["unsw", "cse"], required=True)
ap.add_argument("--processed-dir", default=None)
ap.add_argument("--models-dir", default=None)
ap.add_argument("--fill", choices=["median", "zero"], default="median", help="giá trị thay cho nhóm bị cắt")
args = ap.parse_args()

cfg, mcfg = load_config(), load_models_cfg()
proc = Path(args.processed_dir) if args.processed_dir else resolve_path(cfg, "processed_dir")
mdir = Path(args.models_dir) if args.models_dir else Path(cfg["_root"]) / "models" / args.dataset
rdir = resolve_path(cfg, "reports_dir")

fs = load_feature_set(proc)
feats, groups = fs["features"], fs["groups"]
models = {n: joblib.load(mdir / f"{n}.joblib") for n in BASE_NAMES}
st = joblib.load(mdir / "stack.joblib")
w, b, thr = st["w"], st["b"], st["threshold"]
pi = natural_prevalence(proc, args.dataset)

va = load_split(proc, args.dataset, "val")
te = load_split(proc, args.dataset, "test")
Xva, yva, wva, _ = xyw(va, feats)
Xte, yte, wte, ate = xyw(te, feats)
fill = Xva.median() if args.fill == "median" else pd.Series(0.0, index=Xva.columns)  # median của val, không dùng test

# Baseline (toàn bộ đặc trưng)
def stack_eval(X, y, w_arr, thr, ate, pi):
    F = raw_matrix(models, X)
    z = stack_logit(F, w, b)
    p = 1 / (1 + np.exp(-z))
    return evaluate(y, p, w_arr, thr, ate, pi)

baseline = stack_eval(Xte, yte, wte, thr, ate, pi)
log.info("Baseline DR=%.4f FAR=%.5f macro-F1=%.4f",
         baseline["weighted"]["dr"], baseline["weighted"]["far"], baseline["weighted"]["macro_f1"])

# Xác định các nhóm thực sự có mặt trong feats
group_names_present = [g for g in GROUP_ORDER if any(groups.get(f) == g for f in feats)]
log.info("Nhóm đặc trưng: %s", group_names_present)

ablation_results = {"baseline": baseline, "groups": {}, "fill": args.fill}

for ablate_group in group_names_present:
    ablate_feats = [f for f in feats if groups.get(f) == ablate_group]
    if not ablate_feats:
        continue

    # Tạo bản sao X với cột nhóm bị thay bằng giá trị trung lập (median của val hoặc 0)
    Xte_abl = Xte.copy()
    Xva_abl = Xva.copy()
    for f in ablate_feats:
        Xte_abl[f] = fill[f]
        Xva_abl[f] = fill[f]

    # Chọn ngưỡng lại trên val với nhóm bị cắt
    F_va = raw_matrix(models, Xva_abl)
    z_va = stack_logit(F_va, w, b)
    p_va = 1 / (1 + np.exp(-z_va))
    thr_abl = best_threshold(yva, p_va, wva)

    res_abl = stack_eval(Xte_abl, yte, wte, thr_abl, ate, pi)

    dr_drop = baseline["weighted"]["dr"] - res_abl["weighted"]["dr"]
    far_rise = res_abl["weighted"]["far"] - baseline["weighted"]["far"]
    f1_drop = baseline["weighted"]["macro_f1"] - res_abl["weighted"]["macro_f1"]

    ablation_results["groups"][ablate_group] = {
        "n_features_ablated": len(ablate_feats),
        "features_ablated": ablate_feats,
        "result": res_abl,
        "delta": {
            "dr_drop": float(dr_drop),
            "far_rise": float(far_rise),
            "macro_f1_drop": float(f1_drop),
        }
    }
    log.info("Cắt %-15s | ΔDR=%+.4f ΔFAR=%+.5f ΔF1=%+.4f",
             ablate_group, -dr_drop, far_rise, -f1_drop)

# ── Ablation trên điểm đồng thuận A(x) ──────────────────────────────────────
log.info("Tính ablation điểm đồng thuận A(x)...")
sample_path = proc / args.dataset / "explain_sample.parquet"
sample = pd.read_parquet(sample_path)
phi_stack_cols = [f"phi_{g}" for g in group_names_present if f"phi_{g}" in sample.columns]
gnames = [c[4:] for c in phi_stack_cols]
G_base_list = []
for bname in BASE_NAMES:
    bcols = [f"phi_{bname}_{g}" for g in gnames]
    if all(c in sample.columns for c in bcols):
        G_base_list.append(np.column_stack([sample[c].to_numpy() for c in bcols]))
A_full = agreement_score(G_base_list) if len(G_base_list) >= 2 else np.full(len(sample), np.nan)
agreement_baseline = float(np.nanmean(A_full))
log.info("Agreement A(x) baseline: %.4f", agreement_baseline)

agreement_ablation = {}
for ablate_group in gnames:
    keep_idx = [i for i, g in enumerate(gnames) if g != ablate_group]
    G_base_abl = [G[:, keep_idx] for G in G_base_list]
    A_abl = agreement_score(G_base_abl) if len(G_base_abl) >= 2 else np.full(len(sample), np.nan)
    agreement_ablation[ablate_group] = {
        "agreement_mean": float(np.nanmean(A_abl)),
        "delta": float(np.nanmean(A_abl)) - agreement_baseline,
    }
    log.info("Cắt nhóm %-15s → A(x)=%.4f (Δ=%+.4f)",
             ablate_group, np.nanmean(A_abl), np.nanmean(A_abl) - agreement_baseline)

ablation_results["agreement_baseline"] = agreement_baseline
ablation_results["agreement_ablation"] = agreement_ablation
ablation_results["agreement_ablation_note"] = "chỉ bỏ cột nhóm khỏi vector SHAP đã lưu, không chạy lại mô hình"

write_json(ablation_results, rdir / f"ablation_{args.dataset}.json")
log.info("Đã ghi reports/ablation_%s.json", args.dataset)

# In bảng tóm tắt (cùng định dạng với scripts/13_summary.py)
print(ablation_text(args.dataset, ablation_results))
