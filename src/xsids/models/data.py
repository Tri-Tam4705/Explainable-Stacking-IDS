"""Nạp đặc trưng đã dựng (Bước 2) và tách X, y, trọng số."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from xsids.schema import ATTACK_COL, LABEL_COL, WEIGHT_COL


def load_feature_set(proc_dir) -> dict:
    with open(Path(proc_dir) / "feature_set.json", "r", encoding="utf-8") as f:
        return json.load(f)


def load_split(proc_dir, ds: str, split: str) -> pd.DataFrame:
    return pd.read_parquet(Path(proc_dir) / ds / "features" / f"{split}.parquet")


def xyw(df: pd.DataFrame, features: list[str]):
    return (df[features], df[LABEL_COL].to_numpy(), df[WEIGHT_COL].to_numpy(dtype="float64"),
            df[ATTACK_COL].astype(str).to_numpy())


def natural_prevalence(proc_dir, ds: str) -> float:
    """Tỷ lệ tấn công tự nhiên (có trọng số) của toàn dataset, lấy từ manifest bước 1."""
    with open(Path(proc_dir) / ds / "manifest.json", "r", encoding="utf-8") as f:
        return float(json.load(f)["cleaning"]["attack_ratio_after_weighted"])
