"""Bước 29: baseline ghép đều (trung bình đầu ra thô, trung bình logit) qua tám seed, so với stacking.

Dùng lại mô hình đã huấn luyện của từng seed (models/<bộ>, models/<bộ>_seed1..7), không huấn luyện lại.
Ngưỡng của baseline chọn trên val. Kiểm định ghép cặp như scripts/17_multiseed.py.
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
from scipy.special import expit, logit

import _bootstrap  # noqa: F401
from xsids.config import load_config, resolve_path
from xsids.models.data import load_feature_set, load_split, natural_prevalence, xyw
from xsids.models.evaluate import best_threshold, evaluate
from xsids.models.experiment import paired_summary
from xsids.models.stacking import raw_matrix, stack_logit
from xsids.models.zoo import BASE_NAMES
from xsids.utils import get_logger, write_json

log = get_logger()
ap = argparse.ArgumentParser()
ap.add_argument("--dataset", choices=["unsw", "cse"], required=True)
args = ap.parse_args()
ds = args.dataset
cfg = load_config()
proc, rep = resolve_path(cfg, "processed_dir"), resolve_path(cfg, "reports_dir")
feats = load_feature_set(proc)["features"]
va, te = load_split(proc, ds, "val"), load_split(proc, ds, "test")
Xva, yva, wva, _ = xyw(va, feats)
Xte, yte, wte, ate = xyw(te, feats)
pi = natural_prevalence(proc, ds)
root = Path(cfg["_root"]) / "models"
dirs = [("seed gốc", root / ds)] + [(f"seed{i}", root / f"{ds}_seed{i}") for i in range(1, 8)]


def to_logit(F):
    G = F.copy()
    for k, n in enumerate(BASE_NAMES):
        if n in ("rf", "et"):
            G[:, k] = logit(np.clip(F[:, k], 1e-6, 1 - 1e-6))
    return G


M = ("dr", "far", "macro_f1", "pr_auc")
vals = {k: {m: [] for m in M} for k in ("stack_explainable", "mean_raw", "mean_logit")}
for tag, d in dirs:
    models = {n: joblib.load(d / f"{n}.joblib") for n in BASE_NAMES}
    st = joblib.load(d / "stack.joblib")
    F_va, F_te = raw_matrix(models, Xva), raw_matrix(models, Xte)
    sc = {"stack_explainable": (expit(stack_logit(F_va, st["w"], st["b"])), expit(stack_logit(F_te, st["w"], st["b"])), st["threshold"]),
          "mean_raw": (F_va.mean(1), F_te.mean(1), None),
          "mean_logit": (to_logit(F_va).mean(1), to_logit(F_te).mean(1), None)}
    for k, (sv, stt, thr) in sc.items():
        thr = thr if thr is not None else best_threshold(yva, sv, wva)
        r = evaluate(yte, stt, wte, thr, ate, pi)["weighted"]
        for m in M:
            vals[k][m].append(r[m])
    log.info("%s: stack DR %.4f | mean_raw DR %.4f | mean_logit DR %.4f", tag, vals["stack_explainable"]["dr"][-1],
             vals["mean_raw"]["dr"][-1], vals["mean_logit"]["dr"][-1])

out = {"dataset": ds, "seeds": [t for t, _ in dirs],
       "metrics": {k: {m: {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)), "values": v} for m, v in d.items()} for k, d in vals.items()},
       "stack_vs": {k: {m: paired_summary(vals["stack_explainable"][m], vals[k][m], lower_is_better=(m == "far")) for m in M}
                    for k in ("mean_raw", "mean_logit")}}
write_json(out, rep / f"equal_weight_multiseed_{ds}.json")
