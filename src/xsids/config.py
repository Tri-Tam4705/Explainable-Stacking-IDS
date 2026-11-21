"""Đọc cấu hình YAML và xác định đường dẫn tương đối theo thư mục gốc dự án."""
from __future__ import annotations

from pathlib import Path

import yaml


def project_root() -> Path:
    # src/xsids/config.py -> parents[2] là thư mục gốc dự án
    return Path(__file__).resolve().parents[2]


def load_config(path=None) -> dict:
    root = project_root()
    path = Path(path) if path else root / "configs" / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = str(root)
    return cfg


def resolve_path(cfg: dict, key: str) -> Path:
    p = Path(cfg["paths"][key])
    return p if p.is_absolute() else Path(cfg["_root"]) / p


def dataset_csv_path(cfg: dict, key: str, raw_dir_override=None) -> Path:
    """Đường dẫn CSV của một dataset.

    Ưu tiên: --raw-dir (nếu có) > datasets.<key>.path trong config > raw_dir/file.
    """
    ds = cfg["datasets"][key]
    if raw_dir_override is None and ds.get("path"):
        p = Path(ds["path"])
        return p if p.is_absolute() else Path(cfg["_root"]) / p
    base = Path(raw_dir_override) if raw_dir_override else resolve_path(cfg, "raw_dir")
    p = base / ds["file"]
    # bản clone từ GitHub chỉ có file nén .csv.gz (Git LFS); pandas tự giải nén khi đọc
    gz = p.with_name(p.name + ".gz")
    return gz if not p.exists() and gz.exists() else p
