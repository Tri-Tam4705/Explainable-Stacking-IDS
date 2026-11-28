"""Kiểm toán dữ liệu thô và ghi báo cáo JSON + Markdown vào reports/."""
import argparse
import _bootstrap  # noqa: F401
from xsids.audit import audit_dataframe, render_markdown
from xsids.config import dataset_csv_path, load_config, resolve_path
from xsids.data.labels import add_family
from xsids.data.loader import read_netflow_csv
from xsids.utils import ensure_dir, get_logger, write_json

log = get_logger()
ap = argparse.ArgumentParser()
ap.add_argument("--dataset", choices=["unsw", "cse"], required=True)
ap.add_argument("--raw-dir", default=None, help="thư mục chứa CSV (mặc định data/raw)")
ap.add_argument("--sample-frac", type=float, default=None)
ap.add_argument("--tag", default="", help="hậu tố tên báo cáo, vd. synthetic")
ap.add_argument("--config", default=None)
args = ap.parse_args()

cfg = load_config(args.config)
ds = cfg["datasets"][args.dataset]
csv_path = dataset_csv_path(cfg, args.dataset, args.raw_dir)
frac = args.sample_frac if args.sample_frac is not None else ds["sample_frac"]

df, info = read_netflow_csv(csv_path, cfg["columns"]["label_source"],
                            cfg["columns"]["attack_source"], sample_frac=frac, seed=cfg["seed"],
                            compact=ds.get("compact", False))
df, fam = add_family(df, cfg["families"])
rep = audit_dataframe(df, cfg["audit"], fam)
rep["load"] = info

out = ensure_dir(resolve_path(cfg, "reports_dir"))
stem = f"audit_{args.dataset}" + (f"_{args.tag}" if args.tag else "")
write_json(rep, out / f"{stem}.json")
(out / f"{stem}.md").write_text(render_markdown(rep, ds["name"]), encoding="utf-8")
log.info("Đã ghi %s.json và %s.md trong %s", stem, stem, out)
