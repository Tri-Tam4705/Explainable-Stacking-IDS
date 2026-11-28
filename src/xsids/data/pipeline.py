"""Quy trình bước 1: nạp -> gom họ -> làm sạch -> chia tập -> lưu parquet."""
from __future__ import annotations

from xsids.data.cleaning import clean_dataset
from xsids.data.labels import add_family
from xsids.data.loader import read_netflow_csv
from xsids.data.split import save_splits, three_way_split
from xsids.utils import get_logger

log = get_logger()


def prepare_dataset(csv_path, out_dir, cfg: dict, name: str = "", sample_frac: float | None = None,
                    compact: bool = False) -> dict:
    cols = cfg["columns"]
    seed = cfg["seed"]
    frac = 1.0 if sample_frac is None else sample_frac

    df, load_info = read_netflow_csv(
        csv_path, cols["label_source"], cols["attack_source"], sample_frac=frac, seed=seed, compact=compact
    )
    df, fam_info = add_family(df, cfg["families"])
    df, clean_rep = clean_dataset(df, cfg["cleaning"])

    sp = cfg["split"]
    parts = three_way_split(df, sp["test_size"], sp["val_size"], seed, sp["min_stratum_size"])
    meta = {
        "dataset": name,
        "seed": seed,
        "split_config": sp,
        "load": load_info,
        "family": fam_info,
        "cleaning": clean_rep,
        "feature_columns": clean_rep["features_final"],
    }
    manifest = save_splits(df, parts, out_dir, meta)
    log.info("Đã lưu train/val/test vào %s: %s", out_dir,
             {k: v["rows"] for k, v in manifest["summary"].items()})
    log.info("Tỷ lệ tấn công có trọng số theo tập: %s",
             {k: round(100 * v.get("weighted_attack_ratio", float("nan")), 2)
              for k, v in manifest["summary"].items()})
    return manifest
