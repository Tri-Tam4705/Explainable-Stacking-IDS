"""Nạp CSV NetFlow v2 theo từng khối, chuẩn hóa tên cột nhãn và kiểu dữ liệu."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from xsids.schema import (
    ALL_FEATURES,
    ATTACK_COL,
    IP_COLS,
    LABEL_COL,
    TIME_META_COLS,
)
from xsids.utils import get_logger

log = get_logger()


def read_netflow_csv(
    path,
    label_source: str = "Label",
    attack_source: str = "Attack",
    sample_frac: float = 1.0,
    seed: int = 42,
    chunksize: int = 500_000,
    keep_ip: bool = False,
):
    """Đọc file CSV, trả về (df, info).

    - Chỉ đọc các cột cần thiết (bỏ địa chỉ IP để tiết kiệm RAM).
    - sample_frac < 1: lấy mẫu ĐỀU từng khối nên giữ nguyên tỷ lệ tấn công tự nhiên.
    - Đổi tên cột nhãn về `label` (0/1) và `attack` (chuỗi).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Không thấy file dữ liệu: {path}")

    header = [str(c).strip() for c in pd.read_csv(path, nrows=0).columns]
    want = set(ALL_FEATURES) | set(TIME_META_COLS) | {label_source, attack_source}
    if not keep_ip:
        want -= set(IP_COLS)

    raw_header = pd.read_csv(path, nrows=0).columns.tolist()
    usecols = [raw for raw, clean in zip(raw_header, header) if clean in want]

    missing = sorted(c for c in ALL_FEATURES if c not in header and (keep_ip or c not in IP_COLS))
    extra = sorted(c for c in header if c not in want and c not in IP_COLS)
    if label_source not in header:
        raise KeyError(f"Không thấy cột nhãn '{label_source}'. Các cột có: {header[:10]}...")
    if missing:
        log.warning("Thiếu %d cột đặc trưng so với lược đồ 43: %s", len(missing), missing)
    if extra:
        log.info("Bỏ qua %d cột lạ: %s", len(extra), extra)

    rng = np.random.default_rng(seed)
    parts = []
    n_read = 0
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize, low_memory=False):
        chunk.columns = [str(c).strip() for c in chunk.columns]
        n_read += len(chunk)
        if sample_frac < 1.0:
            chunk = chunk[rng.random(len(chunk)) < sample_frac]
        parts.append(chunk)
    df = pd.concat(parts, ignore_index=True)
    del parts

    df, std_info = _standardize(df, label_source, attack_source)
    info = {
        "file": str(path),
        "rows_in_file": int(n_read),
        "rows_loaded": int(len(df)),
        "sample_frac": sample_frac,
        "missing_feature_cols": missing,
        "ignored_extra_cols": extra,
        **std_info,
    }
    log.info("Đã nạp %s: %d/%d dòng, %d cột", path.name, len(df), n_read, df.shape[1])
    return df, info


def _standardize(df: pd.DataFrame, label_source: str, attack_source: str):
    info = {}

    # --- nhãn nhị phân ---
    lab = pd.to_numeric(df[label_source], errors="coerce")
    if lab.isna().mean() > 0.5:  # nhãn dạng chữ (vd. "Benign"/"Attack")
        is_benign = df[label_source].astype(str).str.strip().str.lower().eq("benign")
        lab = (~is_benign).astype("int8")
    bad = lab.isna()
    info["rows_dropped_bad_label"] = int(bad.sum())
    if bad.any():
        df = df.loc[~bad].copy()
        lab = lab.loc[~bad]
    df[label_source] = lab.astype("int8")

    # --- tên tấn công ---
    if attack_source in df.columns:
        att = df[attack_source].astype(str).str.strip()
    else:
        log.warning("Không có cột '%s': suy ra từ nhãn nhị phân", attack_source)
        att = pd.Series(np.where(df[label_source] == 0, "Benign", "Attack"), index=df.index)
    df[attack_source] = att

    df = df.rename(columns={label_source: LABEL_COL, attack_source: ATTACK_COL})

    # --- ép kiểu số cho đặc trưng (giá trị rác thành NaN, bước làm sạch sẽ đếm) ---
    for c in df.columns:
        if c in (LABEL_COL, ATTACK_COL):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.reset_index(drop=True), info
