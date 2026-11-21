"""Xem nhanh một file CSV NetFlow: cột có/thiếu/lạ, kiểu dữ liệu, tên lớp tấn công thật."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

from xsids.schema import ALL_FEATURES, IP_COLS, TIME_META_COLS


def peek_csv(path, label_source: str = "Label", attack_source: str = "Attack",
             nrows: int = 100_000, scan_attack: bool = False, chunksize: int = 1_000_000) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Không thấy file: {path}")

    head = pd.read_csv(path, nrows=nrows, low_memory=False)
    cols = [str(c).strip() for c in head.columns]
    head.columns = cols

    expected = set(ALL_FEATURES)
    out = {
        "file": str(path),
        "size_mb": round(path.stat().st_size / 1e6, 1),
        "n_columns": len(cols),
        "columns": cols,
        "expected_missing": [c for c in ALL_FEATURES if c not in cols],
        "extra_columns": [c for c in cols if c not in expected and c not in TIME_META_COLS
                          and c not in (label_source, attack_source)],
        "time_columns_present": [c for c in TIME_META_COLS if c in cols],
        "has_label_col": label_source in cols,
        "has_attack_col": attack_source in cols,
        "non_numeric_columns": [c for c in cols if c not in IP_COLS and c != attack_source
                                and not pd.api.types.is_numeric_dtype(head[c])],
        "sample_rows": head.head(3).to_dict(orient="records"),
    }
    if label_source in cols:
        out["label_values_in_head"] = {str(k): int(v) for k, v in head[label_source].value_counts(dropna=False).items()}
    if attack_source in cols:
        out["attack_values_in_head"] = {str(k): int(v) for k, v in head[attack_source].value_counts(dropna=False).items()}

    if scan_attack and attack_source in cols:
        cnt: Counter = Counter()
        for chunk in pd.read_csv(path, usecols=[attack_source], chunksize=chunksize):
            cnt.update(chunk[attack_source].astype(str).str.strip().tolist())
        out["attack_values_full_scan"] = dict(cnt.most_common())
    return out


def render_peek(rep: dict) -> str:
    L = [f"File: {rep['file']} ({rep['size_mb']} MB), {rep['n_columns']} cột"]
    L.append(f"Cột nhãn 'Label': {'có' if rep['has_label_col'] else 'KHÔNG CÓ'} | cột 'Attack': {'có' if rep['has_attack_col'] else 'KHÔNG CÓ'}")
    L.append(f"Cột thời gian: {rep['time_columns_present'] or 'không có'}")
    L.append(f"Thiếu so với lược đồ 43 (IP vẫn tính): {rep['expected_missing'] or 'không thiếu'}")
    L.append(f"Cột lạ ngoài lược đồ: {rep['extra_columns'] or 'không có'}")
    L.append(f"Cột không phải số (trừ IP, Attack): {rep['non_numeric_columns'] or 'không có'}")
    L.append(f"Toàn bộ cột: {rep['columns']}")
    if "label_values_in_head" in rep:
        L.append(f"Giá trị Label trong phần đầu: {rep['label_values_in_head']}")
    if "attack_values_in_head" in rep:
        L.append(f"Giá trị Attack trong phần đầu: {rep['attack_values_in_head']}")
    if "attack_values_full_scan" in rep:
        L.append("Giá trị Attack trên TOÀN BỘ file:")
        for k, v in rep["attack_values_full_scan"].items():
            L.append(f"  {k}: {v:,}")
    return "\n".join(L)
