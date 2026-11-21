"""Bước 8: Đánh giá chéo dataset và LOFO (Leave One Feature Out).

Cross-dataset: train trên UNSW → test trên CSE và ngược lại.
  Đo DR, FAR, macro-F1 và PR-AUC để biết mô hình có khái quát hóa không.

LOFO: lần lượt thay từng đặc trưng bằng giá trị trung lập (không huấn luyện lại), đo macro-F1 giảm bao nhiêu.
  Dùng mô hình tốt nhất đơn (lgbm) để tiết kiệm thời gian.
  --fill median (mặc định): median của tập validation; --fill zero: cách cũ (0 lệch phân phối nên thổi phồng mức giảm).
"""
import argparse
from pathlib import Path

import joblib
import numpy as np

import _bootstrap  # noqa: F401
from xsids.config import load_config, resolve_path
from xsids.models.data import load_feature_set, load_split, natural_prevalence, xyw
from xsids.models.evaluate import best_threshold, evaluate
from xsids.models.stacking import raw_matrix, stack_logit
from xsids.models.zoo import BASE_NAMES, load_models_cfg, raw_output, to_proba
from xsids.report.summaries import cross_text
from xsids.utils import get_logger, write_json

log = get_logger()
ap = argparse.ArgumentParser()
ap.add_argument("--lofo-model", default="lgbm", choices=BASE_NAMES, help="mô hình dùng cho LOFO")
ap.add_argument("--processed-dir", default=None)
ap.add_argument("--models-dir-unsw", default=None)
ap.add_argument("--models-dir-cse", default=None)
ap.add_argument("--fill", choices=["median", "zero"], default="median", help="giá trị thay cho đặc trưng bị loại")
args = ap.parse_args()

cfg, mcfg = load_config(), load_models_cfg()
proc = Path(args.processed_dir) if args.processed_dir else resolve_path(cfg, "processed_dir")
root = Path(cfg["_root"])
mdir = {
    "unsw": Path(args.models_dir_unsw) if args.models_dir_unsw else root / "models" / "unsw",
    "cse": Path(args.models_dir_cse) if args.models_dir_cse else root / "models" / "cse",
}
rdir = resolve_path(cfg, "reports_dir")

fs = load_feature_set(proc)
feats, groups = fs["features"], fs["groups"]

datasets = ["unsw", "cse"]
models_all = {ds: {n: joblib.load(mdir[ds] / f"{n}.joblib") for n in BASE_NAMES} for ds in datasets}
stacks_all = {ds: joblib.load(mdir[ds] / "stack.joblib") for ds in datasets}

# ── Cross-dataset evaluation ────────────────────────────────────────────────
log.info("=== Cross-dataset evaluation ===")
cross_results = {}

for train_ds in datasets:
    for test_ds in datasets:
        key = f"train_{train_ds}_test_{test_ds}"
        models_train = models_all[train_ds]
        st = stacks_all[train_ds]
        w, b, thr = st["w"], st["b"], st["threshold"]

        # Dùng val của train_ds để chọn ngưỡng
        va_train = load_split(proc, train_ds, "val")
        Xva, yva, wva, _ = xyw(va_train, feats)
        F_va = raw_matrix(models_train, Xva)
        z_va = stack_logit(F_va, w, b)
        p_va = 1 / (1 + np.exp(-z_va))
        thr_cross = best_threshold(yva, p_va, wva)

        # Test trên test_ds
        te_test = load_split(proc, test_ds, "test")
        Xte, yte, wte, ate = xyw(te_test, feats)
        pi = natural_prevalence(proc, test_ds)

        F_te = raw_matrix(models_train, Xte)
        z_te = stack_logit(F_te, w, b)
        p_te = 1 / (1 + np.exp(-z_te))
        res = evaluate(yte, p_te, wte, thr_cross, ate, pi)
        cross_results[key] = res

        tag = "✓ in-domain" if train_ds == test_ds else "← cross"
        log.info("[%s] Train=%s Test=%s | DR=%.4f FAR=%.5f F1=%.4f PR-AUC=%.4f %s",
                 key, train_ds, test_ds,
                 res["weighted"]["dr"], res["weighted"]["far"],
                 res["weighted"]["macro_f1"], res["weighted"]["pr_auc"], tag)

# ── LOFO: Leave One Feature Out ─────────────────────────────────────────────
log.info("=== LOFO (model: %s) ===", args.lofo_model)
lofo_results = {}

for ds in datasets:
    log.info("LOFO dataset: %s", ds)
    model_lofo = models_all[ds][args.lofo_model]
    va = load_split(proc, ds, "val")
    te = load_split(proc, ds, "test")
    Xva, yva, wva, _ = xyw(va, feats)
    Xte, yte, wte, ate = xyw(te, feats)
    pi = natural_prevalence(proc, ds)

    fill = Xva.median() if args.fill == "median" else None  # median của val, không dùng test

    # Baseline LOFO
    p_va_base = to_proba(args.lofo_model, raw_output(args.lofo_model, model_lofo, Xva))
    thr_base = best_threshold(yva, p_va_base, wva)
    p_te_base = to_proba(args.lofo_model, raw_output(args.lofo_model, model_lofo, Xte))
    res_base = evaluate(yte, p_te_base, wte, thr_base, ate, pi)
    f1_base = res_base["weighted"]["macro_f1"]
    log.info("  Baseline %s F1=%.4f", ds, f1_base)

    lofo_ds = {"baseline_macro_f1": f1_base, "features": {}}
    for feat in feats:
        # Thay cột đó bằng median của val (hoặc 0 nếu --fill zero)
        v = 0.0 if fill is None else float(fill[feat])
        Xva_lo = Xva.copy(); Xva_lo[feat] = v
        Xte_lo = Xte.copy(); Xte_lo[feat] = v
        p_va_lo = to_proba(args.lofo_model, raw_output(args.lofo_model, model_lofo, Xva_lo))
        thr_lo = best_threshold(yva, p_va_lo, wva)
        p_te_lo = to_proba(args.lofo_model, raw_output(args.lofo_model, model_lofo, Xte_lo))
        res_lo = evaluate(yte, p_te_lo, wte, thr_lo, ate, pi)
        f1_lo = res_lo["weighted"]["macro_f1"]
        drop = f1_base - f1_lo
        lofo_ds["features"][feat] = {
            "macro_f1_without": float(f1_lo),
            "f1_drop": float(drop),
            "group": groups.get(feat, "?"),
        }
    # Sắp xếp giảm dần theo f1_drop
    lofo_ds["features"] = dict(
        sorted(lofo_ds["features"].items(), key=lambda x: x[1]["f1_drop"], reverse=True)
    )
    lofo_results[ds] = lofo_ds


# ── Tổng hợp ──────────────────────────────────────────────────────────────────
result = {
    "cross_dataset": cross_results,
    "lofo": lofo_results,
    "fill": args.fill,
}
write_json(result, rdir / "cross_lofo.json")
log.info("Đã ghi reports/cross_lofo.json")

# Bảng tóm tắt cross-dataset và LOFO (cùng định dạng với scripts/13_summary.py)
print(cross_text(result))
