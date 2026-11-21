"""Xem nhanh file CSV thật: tên cột, tên lớp tấn công. Chạy TRƯỚC khi kiểm toán/làm sạch."""
import argparse

import _bootstrap  # noqa: F401
from xsids.config import dataset_csv_path, load_config, resolve_path
from xsids.data.peek import peek_csv, render_peek
from xsids.utils import write_json

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", choices=["unsw", "cse"], required=True)
ap.add_argument("--raw-dir", default=None)
ap.add_argument("--nrows", type=int, default=100_000, help="số dòng đầu để xem nhanh")
ap.add_argument("--scan-attack", action="store_true", help="đọc cả cột Attack của toàn file để đếm tên lớp")
ap.add_argument("--config", default=None)
args = ap.parse_args()

cfg = load_config(args.config)
path = dataset_csv_path(cfg, args.dataset, args.raw_dir)
rep = peek_csv(path, cfg["columns"]["label_source"], cfg["columns"]["attack_source"],
               nrows=args.nrows, scan_attack=args.scan_attack)
print(render_peek(rep))
write_json(rep, resolve_path(cfg, "reports_dir") / f"peek_{args.dataset}.json")
print(f"\nĐã ghi reports/peek_{args.dataset}.json")
