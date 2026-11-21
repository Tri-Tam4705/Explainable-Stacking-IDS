"""Gom tên tấn công thô thành họ (family) bằng regex trong cấu hình."""
from __future__ import annotations

import re

import pandas as pd

from xsids.schema import ATTACK_COL, FAMILY_COL, LABEL_COL


def add_family(df: pd.DataFrame, families: dict[str, str], default: str = "Other"):
    """Thêm cột `family`. Trả về (df, info). Không cần biết trước tên lớp thật."""
    patterns = [(fam, re.compile(p, re.IGNORECASE)) for fam, p in families.items()]
    names = df[ATTACK_COL].astype(str).unique().tolist()

    mapping = {}
    for name in names:
        fam = default
        for f, pat in patterns:
            if pat.search(name):
                fam = f
                break
        mapping[name] = fam

    df[FAMILY_COL] = df[ATTACK_COL].astype(str).map(mapping)

    is_benign_label = df[LABEL_COL] == 0
    is_benign_family = df[FAMILY_COL] == "Benign"
    inconsistent = int((is_benign_label != is_benign_family).sum())

    info = {
        "mapping": mapping,
        "unmapped_to_other": sorted(n for n, f in mapping.items() if f == default),
        "label_family_inconsistent_rows": inconsistent,
    }
    return df, info
