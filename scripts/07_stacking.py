"""Bước 4: stacking out-of-fold với meta-learner tuyến tính; so với 4 mô hình đơn."""
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np

import _bootstrap  # noqa: F401
from xsids.config import load_config, resolve_path
from xsids.models.data import load_feature_set, load_split, natural_prevalence, xyw
from xsids.models.evaluate import best_threshold, evaluate
from xsids.models.stacking import fit_meta, oof_raw, raw_matrix, stack_logit
from xsids.models.zoo import BASE_NAMES, get_params, load_models_cfg
from xsids.report.summaries import models_text
from xsids.utils import get_logger, write_json

log = get_logger()
ap = argparse.ArgumentParser()
ap.add_argument("--dataset", choices=["unsw", "cse"], required=True)
ap.add_argument("--fast", action="store_true")
ap.add_argument("--processed-dir", default=None)
ap.add_argument("--models-dir", default=None)
ap.add_argument("--tag", default="")
ap.add_argument("--seed", type=int, default=None, help="ghi đè seed trong configs/models.yaml (dùng cho chạy nhiều seed)")
args = ap.parse_args()

cfg, mcfg = load_config(), load_models_cfg()
proc = Path(args.processed_dir) if args.processed_dir else resolve_path(cfg, "processed_dir")
mdir = Path(args.models_dir) if args.models_dir else Path(cfg["_root"]) / "models" / args.dataset
feats = load_feature_set(proc)["features"]
seed, n_jobs, folds = (args.seed if args.seed is not None else mcfg["seed"]), mcfg["n_jobs"], mcfg["cv_folds"]
params = get_params(mcfg, args.fast)

tr = load_split(proc, args.dataset, "train")
va, te = load_split(proc, args.dataset, "val"), load_split(proc, args.dataset, "test")
Xtr, ytr, _, atr = xyw(tr, feats)
Xva, yva, wva, _ = xyw(va, feats)
Xte, yte, wte, ate = xyw(te, feats)
pi = natural_prevalence(proc, args.dataset)
models = {n: joblib.load(mdir / f"{n}.joblib") for n in BASE_NAMES}  # đã huấn luyện ở bước 3 trên toàn train

F_oof = np.zeros((len(Xtr), len(BASE_NAMES)))
for k, n in enumerate(BASE_NAMES):
    t = time.time()
    F_oof[:, k] = oof_raw(n, params[n], Xtr, ytr, atr, folds, seed, n_jobs)
    log.info("OOF %-4s %.0fs", n, time.time() - t)

w, b, _ = fit_meta(F_oof, ytr, mcfg["meta"]["C"])
log.info("Meta-learner (đầu ra thô): w = %s, b = %.4f", {n: round(float(v), 4) for n, v in zip(BASE_NAMES, w)}, b)

F_va, F_te = raw_matrix(models, Xva), raw_matrix(models, Xte)
z_va, z_te = stack_logit(F_va, w, b), stack_logit(F_te, w, b)
s_va, s_te = 1 / (1 + np.exp(-z_va)), 1 / (1 + np.exp(-z_te))

res = {
    "stack_explainable": evaluate(yte, s_te, wte, best_threshold(yva, s_va, wva), ate, pi),
}
stem_b = f"baselines_{args.dataset}" + (f"_{args.tag}" if args.tag else "")
base_path = resolve_path(cfg, "reports_dir") / f"{stem_b}.json"
base_res = json.load(open(base_path, encoding="utf-8"))["results"] if base_path.exists() else {}

joblib.dump({"w": w, "b": b, "names": BASE_NAMES, "features": feats, "C": mcfg["meta"]["C"],
             "threshold": res["stack_explainable"]["threshold"]},
            mdir / "stack.joblib")

rows = [("Mô hình đơn tốt nhất (PR-AUC)", None)] + [(k, k) for k in base_res]
table = {**base_res, **res}
best_single = max([n for n in BASE_NAMES if n in base_res], key=lambda n: base_res[n]["weighted"]["pr_auc"], default=None)
md = ["# So sánh mô hình (tập test, có trọng số = tỷ lệ tấn công tự nhiên)\n",
      f"Tỷ lệ tấn công tự nhiên dùng để tính precision cảnh báo: {100*pi:.2f}%\n",
      "| Mô hình | DR | FAR | Precision cảnh báo @ tỷ lệ tự nhiên | macro-F1 | PR-AUC |", "|---|---|---|---|---|---|"]
for name, r in table.items():
    x = r["weighted"]
    md.append(f"| {name}{' (tốt nhất trong 4 mô hình đơn)' if name == best_single else ''} | {x['dr']:.4f} | "
              f"{x['far']:.5f} | {x['precision_at_natural_prevalence']:.4f} | {x['macro_f1']:.4f} | {x['pr_auc']:.4f} |")
stem = f"stacking_{args.dataset}" + (f"_{args.tag}" if args.tag else "")
write_json({"w": {n: float(v) for n, v in zip(BASE_NAMES, w)}, "b": b, "results": res, "fast": args.fast},
           resolve_path(cfg, "reports_dir") / f"{stem}.json")
(resolve_path(cfg, "reports_dir") / f"{stem}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print(models_text(args.dataset, table))
