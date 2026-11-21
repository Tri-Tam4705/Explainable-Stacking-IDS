"""Sinh CSV giả lập có lược đồ NetFlow v2 để thử pipeline (không phải dữ liệu thật)."""
import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from xsids.config import load_config
from xsids.data.synthetic import make_synthetic_netflow
from xsids.utils import ensure_dir, get_logger

log = get_logger()
ap = argparse.ArgumentParser()
ap.add_argument("--dataset", choices=["unsw", "cse"], required=True)
ap.add_argument("--n", type=int, default=100_000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out-dir", default="data/synthetic")
ap.add_argument("--config", default=None)
args = ap.parse_args()

cfg = load_config(args.config)
out_dir = ensure_dir(Path(cfg["_root"]) / args.out_dir)
df = make_synthetic_netflow(args.n, style=args.dataset, seed=args.seed)
path = out_dir / cfg["datasets"][args.dataset]["file"]
df.to_csv(path, index=False)
log.info("Đã ghi %s (%d dòng, tỷ lệ tấn công %.2f%%)", path, len(df), 100 * df["Label"].mean())
