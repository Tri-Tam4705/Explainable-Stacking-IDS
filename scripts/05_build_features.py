"""Bước 2: dựng bộ đặc trưng chung (sàng trên train, giữ phần chung, gộp tương quan)."""
import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from xsids.config import load_config, resolve_path
from xsids.features.build import build_features, load_features_cfg, render_report
from xsids.utils import ensure_dir

ap = argparse.ArgumentParser()
ap.add_argument("--datasets", nargs="+", choices=["unsw", "cse"], default=["unsw", "cse"])
ap.add_argument("--processed-dir", default=None)
ap.add_argument("--include", nargs="*", default=[], help="đặc trưng tùy chọn cho thí nghiệm cắt bỏ, vd. L7_PROTO")
ap.add_argument("--tag", default="")
args = ap.parse_args()

cfg = load_config()
proc = Path(args.processed_dir) if args.processed_dir else resolve_path(cfg, "processed_dir")
fs = build_features(proc, args.datasets, load_features_cfg(), tuple(args.include))
out = ensure_dir(resolve_path(cfg, "reports_dir"))
name = "features" + (f"_{args.tag}" if args.tag else "") + ".md"
(out / name).write_text(render_report(fs), encoding="utf-8")
print(f"Đã ghi reports/{name} và {proc / 'feature_set.json'}")
