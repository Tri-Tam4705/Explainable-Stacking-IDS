"""Lặp lại huấn luyện và đánh giá trên nhiều phép chia train/val/test khác nhau, rồi kiểm định ghép cặp stacking so với voting.

Bước nhiều seed (17_multiseed.py) giữ nguyên phép chia nên độ lệch chuẩn chỉ phản ánh ngẫu nhiên của huấn luyện. Ở đây các
vector đã làm sạch (train + val + test của bước 2) được gộp lại và chia lại 70/10/20 với seed chia khác nhau, cùng quy tắc với
bước 1 (phân tầng theo tên tấn công, cân bằng khối lượng). Vector vẫn là duy nhất nên không có rò rỉ giữa các tập.

Kiểm định: t ghép cặp thông thường, t đã hiệu chỉnh của Nadeau và Bengio (2003) cho các phép chia có tập train chồng
nhau, và Wilcoxon dấu-hạng. Kết luận dựa trên kiểm định đã hiệu chỉnh.

Hạn chế: 25 đặc trưng được giữ nguyên như đã chọn trên tập train gốc, không chọn lại cho từng phép chia.

Dùng:
    python scripts/20_multisplit.py --dataset unsw --splits 1 2 3 4 5 6 7 8 9 10
    python scripts/20_multisplit.py --dataset cse --splits 1 2 --fast
    python scripts/20_multisplit.py --dataset unsw --aggregate-only
Đầu ra: reports/multisplit_<ds>.json. Không ghi đè mô hình hay kết quả chính.
Thời gian: mỗi phép chia bằng bước 3 + 4 (khoảng 3 phút với UNSW, 8 phút với CSE).
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from xsids.config import load_config, resolve_path
from xsids.data.split import three_way_split
from xsids.models.data import cap_train, load_feature_set, load_split, natural_prevalence, xyw
from xsids.models.experiment import METRICS, evaluate_system, fit_system, paired_summary, thresholds
from xsids.models.zoo import get_params, load_models_cfg
from xsids.utils import get_logger, write_json

NAMES = {"rf": "Random Forest", "et": "Extra Trees", "xgb": "XGBoost", "lgbm": "LightGBM", "voting": "Voting",
         "stack_explainable": "Stacking (giải thích được)"}
SHOW = [("dr", "DR"), ("far", "FAR"), ("macro_f1", "macro-F1"), ("pr_auc", "PR-AUC")]

log = get_logger()
ap = argparse.ArgumentParser()
ap.add_argument("--dataset", choices=["unsw", "cse"], required=True)
ap.add_argument("--splits", type=int, nargs="+", default=list(range(1, 11)), help="seed của từng phép chia")
ap.add_argument("--fast", action="store_true")
ap.add_argument("--aggregate-only", action="store_true")
args = ap.parse_args()

cfg, mcfg = load_config(), load_models_cfg()
ds = args.dataset
proc, rdir = resolve_path(cfg, "processed_dir"), resolve_path(cfg, "reports_dir")
out_path = rdir / f"multisplit_{ds}.json"
out = json.load(open(out_path, encoding="utf-8")) if out_path.exists() else {}
if out.get("fast", args.fast) != args.fast or out.get("dataset", ds) != ds:
    out = {}
out.update({"dataset": ds, "fast": args.fast})
runs = out.setdefault("runs", {})

if not args.aggregate_only:
    feats = load_feature_set(proc)["features"]
    seed, n_jobs, folds, C = mcfg["seed"], mcfg["n_jobs"], mcfg["cv_folds"], mcfg["meta"]["C"]
    params, sp = get_params(mcfg, args.fast), cfg["split"]
    pi = natural_prevalence(proc, ds)
    df = pd.concat([load_split(proc, ds, s) for s in ("train", "val", "test")], ignore_index=True)
    log.info("%s: %d vector duy nhất, chia lại với seed %s", ds, len(df), args.splits)
    for k in args.splits:
        t0 = time.time()
        parts = three_way_split(df, sp["test_size"], sp["val_size"], k, sp["min_stratum_size"])
        assert not (set(parts["train"]) & set(parts["test"])) and not (set(parts["val"]) & set(parts["test"]))
        tr = cap_train(df.iloc[parts["train"]].reset_index(drop=True), mcfg["train_cap"].get(ds), seed)
        va, te = df.iloc[parts["val"]], df.iloc[parts["test"]]
        Xtr, ytr, _, atr = xyw(tr, feats)
        Xva, yva, wva, _ = xyw(va, feats)
        Xte, yte, wte, ate = xyw(te, feats)
        system = fit_system(Xtr, ytr, atr, params, seed, n_jobs, folds, C)
        res = evaluate_system(system, thresholds(system, Xva, yva, wva), Xte, yte, wte, ate, pi)
        runs[str(k)] = {"results": res, "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
                        "meta_weights": {n: float(v) for n, v in zip(("rf", "et", "xgb", "lgbm"), system["w"])},
                        "seconds": round(time.time() - t0, 1)}
        s, v = res["stack_explainable"], res["voting"]
        log.info("chia %d (%.0fs): stacking DR %.4f FAR %.5f PR-AUC %.4f | voting DR %.4f FAR %.5f PR-AUC %.4f",
                 k, time.time() - t0, s["dr"], s["far"], s["pr_auc"], v["dr"], v["far"], v["pr_auc"])
        write_json(out, out_path)

if len(runs) < 2:
    raise SystemExit("Cần ít nhất 2 phép chia để tổng hợp.")
keys = sorted(runs, key=int)
agg = {m: {q: {"mean": float(np.mean([runs[k]["results"][m][q] for k in keys])),
               "std": float(np.std([runs[k]["results"][m][q] for k in keys], ddof=1)),
               "values": [float(runs[k]["results"][m][q]) for k in keys]} for q in METRICS} for m in NAMES}
ratio = float(np.mean([runs[k]["n_test"] / runs[k]["n_train"] for k in keys]))  # n_test / n_train cho hiệu chỉnh Nadeau-Bengio
paired = {q: paired_summary([runs[k]["results"]["stack_explainable"][q] for k in keys],
                            [runs[k]["results"]["voting"][q] for k in keys], lower_is_better=(q == "far"),
                            test_train_ratio=ratio) for q in METRICS}
out.update({"splits": [int(k) for k in keys], "metrics": agg, "stack_minus_voting": paired})
write_json(out, out_path)

print(f"\nNHIỀU PHÉP CHIA — {ds.upper()}: trung bình ± độ lệch chuẩn qua {len(keys)} phép chia train/val/test (seed chia {', '.join(keys)})")
print("─" * 100)
print(f"  {'Mô hình':28s}" + "".join(f"{lab:>18s}" for _, lab in SHOW))
for m, lab in NAMES.items():
    cells = [f"{agg[m][q]['mean']:.{5 if q == 'far' else 4}f} ± {agg[m][q]['std']:.{5 if q == 'far' else 4}f}" for q, _ in SHOW]
    print(f"  {lab:28s}" + "".join(f"{c:>18s}" for c in cells))
print(f"\n  Stacking (giải thích được) − Voting, ghép cặp theo phép chia (hai phía; n_test/n_train = {ratio:.2f}):")
for q, lab in SHOW:
    p, d = paired[q], (5 if q == "far" else 4)
    print(f"    {lab:9s} {p['mean_diff']:+.{d}f} ± {p['std_diff']:.{d}f}   p(t) = {p['p_value_paired_t']:.4f}"
          f"   p(t hiệu chỉnh) = {p['p_value_corrected_t']:.4f}   p(Wilcoxon) = {p['p_value_wilcoxon']:.4f}"
          f"   (stacking tốt hơn ở {p['a_better_in']}/{p['n']} phép chia)")
print("  Độ lệch chuẩn ở đây gồm cả ngẫu nhiên của việc chia dữ liệu. 25 đặc trưng giữ nguyên, không chọn lại cho từng phép chia.")
