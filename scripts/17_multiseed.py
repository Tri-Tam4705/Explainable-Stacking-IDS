"""Chạy lại bước 3 và 4 (4 mô hình đơn và stacking) với nhiều seed rồi báo cáo trung bình ± độ lệch chuẩn.

Mục đích: các mô hình chênh nhau ở chữ số thập phân thứ 3–4, nên một lần chạy không đủ để xếp hạng stacking so với từng mô hình đơn.
Mỗi seed lưu mô hình ở models/<ds>_seed<k>/ và báo cáo ở reports/*_seed<k>.json; KHÔNG ghi đè mô hình và kết quả chính
(models/<ds>/, bước SHAP vẫn dùng seed gốc). Seed ảnh hưởng đến mô hình cây và cách chia fold OOF; phép chia train/val/test
của bước 1 giữ nguyên, nên độ lệch chuẩn ở đây chỉ phản ánh tính ngẫu nhiên của huấn luyện, không phải của việc chia dữ liệu.

Dùng:
    python scripts/17_multiseed.py --dataset unsw --seeds 1 2 3 4 5 6 7  # chạy + tổng hợp (seed đã có kết quả thì bỏ qua)
    python scripts/17_multiseed.py --dataset cse --seeds 1 2 3 4 5 6 7
    python scripts/17_multiseed.py --dataset unsw --seeds 1 2 3 --fast   # thử nhanh
    python scripts/17_multiseed.py --dataset unsw --seeds 1 2 3 --aggregate-only   # chỉ tổng hợp lại
Seed nào đã có reports/baselines_<ds>_seed<k>.json và stacking_<ds>_seed<k>.json (ở reports/ hoặc reports/thu_nghiem/)
thì không huấn luyện lại, trừ khi có --force. Chênh lệch Stacking − từng mô hình đơn có kiểm định t ghép cặp và Wilcoxon (cùng phép
chia nên không cần hiệu chỉnh Nadeau-Bengio).
Thời gian ≈ số seed × (thời gian bước 3 + 4): khoảng 3 phút mỗi seed với UNSW, khoảng 2 giờ với CSE (6,2 triệu vector train).
Đầu ra: reports/multiseed_<ds>.json và bảng in ra màn hình. Seed gốc của dự án (42) được thêm vào bảng nếu có kết quả.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
from xsids.config import load_config, resolve_path
from xsids.models.experiment import stack_vs_base
from xsids.utils import write_json

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", choices=["unsw", "cse"], required=True)
ap.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 8)))
ap.add_argument("--force", action="store_true", help="huấn luyện lại cả seed đã có kết quả")
ap.add_argument("--fast", action="store_true")
ap.add_argument("--aggregate-only", action="store_true")
args = ap.parse_args()

cfg = load_config()
root, rdir, S = Path(cfg["_root"]), resolve_path(cfg, "reports_dir"), Path(__file__).resolve().parent
ds = args.dataset


def result_files(tag: str):
    sfx = f"_{tag}" if tag else ""
    bp, sp = rdir / f"baselines_{ds}{sfx}.json", rdir / f"stacking_{ds}{sfx}.json"
    if tag and not (bp.exists() and sp.exists()):  # kết quả từng seed có thể đã được dọn vào reports/thu_nghiem/
        bp, sp = rdir / "thu_nghiem" / bp.name, rdir / "thu_nghiem" / sp.name
    return bp, sp


if not args.aggregate_only:
    for sd in args.seeds:
        if not args.force and all(f.exists() for f in result_files(f"seed{sd}")):
            print(f"Seed {sd}: đã có kết quả, bỏ qua (dùng --force để huấn luyện lại).", flush=True)
            continue
        common = ["--dataset", ds, "--seed", str(sd), "--tag", f"seed{sd}", "--models-dir", str(root / "models" / f"{ds}_seed{sd}")]
        common += ["--fast"] if args.fast else []
        for script in ("06_baselines.py", "07_stacking.py"):
            print(f"\n>>> {script} {' '.join(common)}", flush=True)
            subprocess.run([sys.executable, str(S / script), *common], check=True)

NAMES = {"rf": "Random Forest", "et": "Extra Trees", "xgb": "XGBoost", "lgbm": "LightGBM",
         "stack_explainable": "Stacking (giải thích được)"}
runs = {}  # seed -> {model -> weighted metrics}
for tag in [""] + [f"seed{s}" for s in args.seeds]:
    bp, sp = result_files(tag)
    if not (bp.exists() and sp.exists()):
        continue
    res = {**json.load(open(bp, encoding="utf-8"))["results"], **json.load(open(sp, encoding="utf-8"))["results"]}
    runs[tag or "seed gốc"] = {k: res[k]["weighted"] for k in NAMES if k in res}
if len(runs) < 2:
    sys.exit("Cần ít nhất 2 lần chạy để tính độ lệch chuẩn (kiểm tra reports/baselines_*_seed*.json).")

METRICS = [("dr", "DR"), ("far", "FAR"), ("macro_f1", "macro-F1"), ("pr_auc", "PR-AUC")]
agg = {}
for k in NAMES:
    if all(k in r for r in runs.values()):
        agg[k] = {m: {"mean": float(np.mean([r[k][m] for r in runs.values()])),
                      "std": float(np.std([r[k][m] for r in runs.values()], ddof=1)),
                      "values": [float(r[k][m]) for r in runs.values()]} for m, _ in METRICS}
# chênh lệch ghép cặp theo seed: stacking giải thích được trừ từng mô hình đơn
paired = stack_vs_base(agg, [m for m, _ in METRICS])
write_json({"dataset": ds, "seeds": list(runs), "metrics": agg, "stack_vs_base": paired}, rdir / f"multiseed_{ds}.json")

print(f"\nMÔ HÌNH — {ds.upper()}: trung bình ± độ lệch chuẩn qua {len(runs)} lần chạy ({', '.join(runs)})")
print("─" * 100)
print(f"  {'Mô hình':28s}" + "".join(f"{lab:>18s}" for _, lab in METRICS))
for k, v in agg.items():
    cells = []
    for m, _ in METRICS:
        d = 5 if m == "far" else 4
        cells.append(f"{v[m]['mean']:.{d}f} ± {v[m]['std']:.{d}f}")
    print(f"  {NAMES[k]:28s}" + "".join(f"{c:>18s}" for c in cells))
for base, ent in paired.items():
    print(f"\n  Stacking (giải thích được) − {NAMES[base]}, ghép cặp theo seed:")
    for m, lab in METRICS:
        p = ent[m]
        d = 5 if m == "far" else 4
        print(f"    {lab:9s} {p['mean_diff']:+.{d}f} ± {p['std_diff']:.{d}f}   p(t) = {p['p_value_paired_t']:.4f}   p(Wilcoxon) = {p['p_value_wilcoxon']:.4f}"
              f"   (stacking tốt hơn ở {p['a_better_in']}/{p['n']} seed)")
if paired:
    print("  Nếu |chênh lệch trung bình| nhỏ hơn độ lệch chuẩn thì chưa đủ cơ sở xếp hạng hai mô hình.")
