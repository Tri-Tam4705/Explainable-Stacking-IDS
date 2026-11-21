"""Làm sạch, chia train/val/test và lưu parquet vào data/processed/<dataset>/."""
import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from xsids.config import dataset_csv_path, load_config, resolve_path
from xsids.data.pipeline import prepare_dataset
from xsids.utils import write_json

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", choices=["unsw", "cse"], required=True)
ap.add_argument("--raw-dir", default=None)
ap.add_argument("--processed-dir", default=None)
ap.add_argument("--sample-frac", type=float, default=None)
ap.add_argument("--tag", default="")
ap.add_argument("--config", default=None)
args = ap.parse_args()

cfg = load_config(args.config)
ds = cfg["datasets"][args.dataset]
csv_path = dataset_csv_path(cfg, args.dataset, args.raw_dir)
proc_dir = Path(args.processed_dir) if args.processed_dir else resolve_path(cfg, "processed_dir")
frac = args.sample_frac if args.sample_frac is not None else ds["sample_frac"]

manifest = prepare_dataset(csv_path, proc_dir / args.dataset, cfg, ds["name"], frac)
stem = f"prepare_{args.dataset}" + (f"_{args.tag}" if args.tag else "")
write_json(manifest, resolve_path(cfg, "reports_dir") / f"{stem}.json")
