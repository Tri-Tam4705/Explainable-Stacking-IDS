"""Bước 3: huấn luyện 4 mô hình đơn + soft voting, chọn ngưỡng trên val, đánh giá trên test."""
import argparse
import time
from pathlib import Path

import joblib
import numpy as np

import _bootstrap  # noqa: F401
from xsids.config import load_config, resolve_path
from xsids.models.data import cap_train, load_feature_set, load_split, natural_prevalence, xyw
from xsids.models.evaluate import best_threshold, evaluate
from xsids.models.zoo import BASE_NAMES, get_params, load_models_cfg, make_model, raw_output, to_proba
from xsids.report.summaries import models_text
from xsids.utils import ensure_dir, get_logger, write_json

log = get_logger()
ap = argparse.ArgumentParser()
ap.add_argument("--dataset", choices=["unsw", "cse"], required=True)
ap.add_argument("--fast", action="store_true", help="tham số nhỏ để chạy thử nhanh")
ap.add_argument("--processed-dir", default=None)
ap.add_argument("--models-dir", default=None)
ap.add_argument("--tag", default="")
ap.add_argument("--seed", type=int, default=None, help="ghi đè seed trong configs/models.yaml (dùng cho chạy nhiều seed)")
args = ap.parse_args()

cfg, mcfg = load_config(), load_models_cfg()
proc = Path(args.processed_dir) if args.processed_dir else resolve_path(cfg, "processed_dir")
mdir = ensure_dir(Path(args.models_dir) if args.models_dir else Path(cfg["_root"]) / "models" / args.dataset)
feats = load_feature_set(proc)["features"]
seed, n_jobs = (args.seed if args.seed is not None else mcfg["seed"]), mcfg["n_jobs"]
params = get_params(mcfg, args.fast)

tr = cap_train(load_split(proc, args.dataset, "train"), mcfg["train_cap"].get(args.dataset), seed)
va, te = load_split(proc, args.dataset, "val"), load_split(proc, args.dataset, "test")
Xtr, ytr, _, _ = xyw(tr, feats)
Xva, yva, wva, _ = xyw(va, feats)
Xte, yte, wte, ate = xyw(te, feats)
pi = natural_prevalence(proc, args.dataset)
log.info("%s: train %d vector (%.2f%% tấn công), val %d, test %d, tỷ lệ tự nhiên %.2f%%",
         args.dataset, len(tr), 100 * ytr.mean(), len(va), len(te), 100 * pi)

res, p_val, p_te = {}, {}, {}
for name in BASE_NAMES:
    t = time.time()
    m = make_model(name, params[name], seed, n_jobs).fit(Xtr, ytr)
    joblib.dump(m, mdir / f"{name}.joblib")
    p_val[name] = to_proba(name, raw_output(name, m, Xva))
    p_te[name] = to_proba(name, raw_output(name, m, Xte))
    thr = best_threshold(yva, p_val[name], wva)
    res[name] = evaluate(yte, p_te[name], wte, thr, ate, pi)
    log.info("%-4s %.0fs | DR %.4f FAR %.5f macro-F1 %.4f PR-AUC %.4f", name, time.time() - t,
             res[name]["weighted"]["dr"], res[name]["weighted"]["far"],
             res[name]["weighted"]["macro_f1"], res[name]["weighted"]["pr_auc"])

vote_val = np.mean([p_val[n] for n in BASE_NAMES], axis=0)
vote_te = np.mean([p_te[n] for n in BASE_NAMES], axis=0)
res["voting"] = evaluate(yte, vote_te, wte, best_threshold(yva, vote_val, wva), ate, pi)

stem = f"baselines_{args.dataset}" + (f"_{args.tag}" if args.tag else "")
write_json({"fast": args.fast, "train_vectors": len(tr), "features": feats, "results": res},
           resolve_path(cfg, "reports_dir") / f"{stem}.json")
log.info("Đã ghi reports/%s.json và lưu mô hình vào %s", stem, mdir)
print(models_text(args.dataset, res))
