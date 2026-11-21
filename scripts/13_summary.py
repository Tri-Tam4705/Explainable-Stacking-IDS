"""Bước 9: tổng hợp kết quả toàn hệ thống thành bảng dễ so sánh và các biểu đồ.

Chỉ đọc các file trong reports/ (và data/processed/<ds>/lime_sample.parquet cho một hình), không huấn luyện hay tính lại gì,
nên chạy trong vài giây sau khi các bước trước đã xong. Thiếu kết quả của bước nào thì phần đó được bỏ qua.

Đầu ra: in ra màn hình, ghi reports/summary.txt và các hình reports/figures/fig_*.png
"""
import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from xsids.config import load_config, resolve_path
from xsids.report.plots import make_all
from xsids.report.summaries import load_reports, render_all
from xsids.utils import ensure_dir, get_logger

log = get_logger()
ap = argparse.ArgumentParser()
ap.add_argument("--datasets", nargs="+", choices=["unsw", "cse"], default=["unsw", "cse"])
ap.add_argument("--reports-dir", default=None)
ap.add_argument("--processed-dir", default=None)
ap.add_argument("--no-figures", action="store_true", help="chỉ in bảng, không vẽ hình")
ap.add_argument("--no-text-file", action="store_true", help="không ghi reports/summary.txt")
args = ap.parse_args()

cfg = load_config()
rdir = Path(args.reports_dir) if args.reports_dir else resolve_path(cfg, "reports_dir")
proc = Path(args.processed_dir) if args.processed_dir else resolve_path(cfg, "processed_dir")

rep = load_reports(rdir, args.datasets)
missing = [f"{k}_{ds}" for k in ("baselines", "stacking", "explain", "stability", "ablation", "lime")
           for ds in args.datasets if not rep[k].get(ds)] + ([] if rep["cross_lofo"] else ["cross_lofo"])
if missing:
    log.warning("Thiếu kết quả của: %s (các phần tương ứng sẽ bị bỏ qua)", ", ".join(missing))

text = render_all(rep)
print(text)
if not args.no_text_file:
    (ensure_dir(rdir) / "summary.txt").write_text(text, encoding="utf-8")
    log.info("Đã ghi %s", rdir / "summary.txt")

if not args.no_figures:
    figs = make_all(rep, ensure_dir(rdir / "figures"), proc, log=lambda m: log.warning(m.strip()))
    print("Biểu đồ đã tạo trong", rdir / "figures")
    for p in figs:
        print("  -", p.name)
