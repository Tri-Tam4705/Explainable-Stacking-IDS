"""Làm sạch dữ liệu TRƯỚC khi chia tập.

Chỉ dùng các phép biến đổi không học từ thống kê toàn cục: xóa cột theo quy tắc, loại dòng
giá trị không hợp lệ, gộp các dòng có vector đặc trưng giống hệt. Mọi thứ cần "học" (scale,
điền thiếu, chọn đặc trưng) để sang bước sau và chỉ fit trên tập train.

Gộp trùng lặp KHÔNG vứt thông tin tần suất: mỗi vector duy nhất giữ lại trọng số n_orig (số dòng
gốc). Nhờ vậy chia tập theo vector duy nhất (không rò rỉ) mà vẫn tính được các chỉ số ở tỷ lệ
tấn công tự nhiên bằng cách tính có trọng số.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from xsids.schema import (
    ATTACK_COL,
    FAMILY_COL,
    HARD_DROP_COLS,
    LABEL_COL,
    MINORITY_COL,
    TIME_META_COLS,
    WEIGHT_COL,
)
from xsids.utils import get_logger

log = get_logger()

NON_FEATURE_COLS = {LABEL_COL, ATTACK_COL, FAMILY_COL, WEIGHT_COL, MINORITY_COL, *TIME_META_COLS}
FLOAT32_MAX = 3.0e38


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def drop_hard_columns(df: pd.DataFrame):
    """Bỏ định danh (IP, cổng) và TTL: đặc trưng 'lối tắt' phụ thuộc môi trường thu thập."""
    dropped = [c for c in HARD_DROP_COLS if c in df.columns]
    return df.drop(columns=dropped), dropped


def invalid_value_mask(df: pd.DataFrame, cols: list[str], max_abs: float | None):
    """Dòng có giá trị không hữu hạn (inf/NaN) hoặc vượt ngưỡng vật lý (giá trị đệm kiểu 1.8e308)."""
    bad_nonfinite = np.zeros(len(df), dtype=bool)
    bad_range = np.zeros(len(df), dtype=bool)
    nf_by_col, range_by_col = {}, {}
    for c in cols:
        x = df[c].to_numpy(dtype="float64", na_value=np.nan)
        m_nf = ~np.isfinite(x)
        with np.errstate(invalid="ignore"):
            m_rg = (np.abs(x) > max_abs) & ~m_nf if max_abs else np.zeros(len(x), dtype=bool)
        if m_nf.any():
            nf_by_col[c] = int(m_nf.sum())
            bad_nonfinite |= m_nf
        if m_rg.any():
            range_by_col[c] = int(m_rg.sum())
            bad_range |= m_rg
    return bad_nonfinite, bad_range, nf_by_col, range_by_col


def nonfinite_mask(df: pd.DataFrame, cols: list[str]):
    """Giữ lại cho kiểm toán: (mask dòng không hữu hạn, đếm theo cột)."""
    nf, _, by_col, _ = invalid_value_mask(df, cols, None)
    return nf, by_col


def label_conflicts(df: pd.DataFrame, cols: list[str]):
    """Nhóm dòng có đặc trưng GIỐNG HỆT nhưng nhãn nhị phân khác nhau."""
    h = pd.util.hash_pandas_object(df[cols], index=False)
    n_label = df[LABEL_COL].groupby(h.to_numpy()).transform("nunique")
    mask = (n_label > 1).to_numpy()
    n_groups = int(pd.Series(h.to_numpy()[mask]).nunique()) if mask.any() else 0
    return mask, n_groups


def collapse_duplicates(df: pd.DataFrame, feats: list[str], policy: str = "majority"):
    """Gộp các dòng có vector đặc trưng giống hệt thành MỘT dòng đại diện kèm trọng số.

    policy="majority": nhãn đại diện là nhãn chiếm đa số trong nhóm (hòa thì chọn "tấn công", an
        toàn hơn trong bảo mật); phần nhãn thiểu số được ghi ở cột n_minority (nhiễu nhãn).
    policy="drop": bỏ hẳn các nhóm có xung đột nhãn.
    Tên tấn công đại diện là tên xuất hiện NHIỀU NHẤT trong các dòng cùng nhãn đại diện của vector
    (hòa thì lấy tên theo thứ tự xuất hiện đầu tiên), không phụ thuộc thứ tự dòng trong file.
    """
    h = pd.util.hash_pandas_object(df[feats], index=False).to_numpy()
    y = df[LABEL_COL].to_numpy()
    grp = pd.Series(y.astype("int64")).groupby(h, sort=False)
    n_total = grp.transform("size").to_numpy()
    n_attack = grp.transform("sum").to_numpy()
    maj = (2 * n_attack >= n_total).astype("int8")
    n_agree = np.where(maj == 1, n_attack, n_total - n_attack)
    n_minority = (n_total - n_agree).astype("int32")
    conflict = n_minority > 0
    minority_mask = y != maj

    a_codes, _ = pd.factorize(df[ATTACK_COL].astype(str))
    take = y == maj
    if policy == "drop":
        take &= ~conflict
    pos = np.flatnonzero(take)

    t = pd.DataFrame({"h": h[pos], "a": a_codes[pos], "y": y[pos]})
    cnt = t.groupby(["h", "a", "y"], sort=False).size().rename("c").reset_index()
    cnt = cnt.sort_values(["h", "c", "a"], ascending=[True, False, True], kind="stable")
    best = cnt.drop_duplicates("h", keep="first")
    chosen = pd.Series(best["a"].to_numpy(), index=best["h"].to_numpy()).reindex(h[pos]).to_numpy()
    pos = pos[a_codes[pos] == chosen]
    first = ~pd.Series(h[pos]).duplicated(keep="first").to_numpy()
    keep = pos[first]

    out = df.iloc[keep].reset_index(drop=True)
    out[WEIGHT_COL] = n_total[keep].astype("int32")
    out[MINORITY_COL] = n_minority[keep]

    # độ mơ hồ của tên tấn công: cùng một vector xuất hiện dưới >= 2 tên tấn công khác nhau
    att = cnt[cnt["y"] == 1]
    n_names = att.groupby("h", sort=False)["a"].transform("size")
    amb = att[n_names > 1]

    n_unique = int(pd.Series(h).nunique())
    info = {
        "rows_in": int(len(df)),
        "unique_vectors": n_unique,
        "duplicate_rows_removed": int(len(df) - n_unique),
        "conflict_rows": int(conflict.sum()),
        "conflict_groups": int(pd.Series(h[conflict]).nunique()) if conflict.any() else 0,
        "conflict_rows_by_attack": {str(k): int(v) for k, v in
                                    df.loc[conflict, ATTACK_COL].value_counts().items()},
        "minority_rows_total": int(minority_mask.sum()),
        "minority_rows_by_attack": {str(k): int(v) for k, v in
                                    df.loc[minority_mask, ATTACK_COL].value_counts().items()},
        "attack_name_ambiguous_groups": int(amb["h"].nunique()),
        "attack_name_ambiguous_rows": int(amb["c"].sum()),
        "attack_rows_in_majority_attack_vectors": int(att["c"].sum()),
        "policy": policy,
        "rows_out": int(len(out)),
    }
    return out, info


def downcast_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Giảm bộ nhớ an toàn: số nguyên -> int nhỏ nhất, số thực -> float32 CHỈ KHI không tràn."""
    for c in cols:
        s = df[c]
        if pd.api.types.is_integer_dtype(s):
            df[c] = pd.to_numeric(s, downcast="integer")
        elif pd.api.types.is_float_dtype(s):
            x = s.to_numpy(dtype="float64", na_value=np.nan)
            finite = x[np.isfinite(x)]
            if len(finite) == 0:
                continue
            fmax = float(np.abs(finite).max())
            if (finite == np.floor(finite)).all() and not np.isnan(x).any() and fmax < 2**62:
                df[c] = pd.to_numeric(s.astype("int64"), downcast="integer")
            elif fmax <= FLOAT32_MAX:
                df[c] = s.astype("float32")
            else:
                log.warning("Cột %s có giá trị %.3g vượt float32: giữ float64", c, fmax)
    return df


def clean_dataset(df: pd.DataFrame, cfg_cleaning: dict):
    """Trả về (df_sạch, báo cáo). df_sạch có thêm cột n_orig và n_minority."""
    rep = {"rows_before": int(len(df))}
    rep["attack_ratio_before"] = float(df[LABEL_COL].mean())
    rep["attack_counts_before"] = {str(k): int(v) for k, v in df[ATTACK_COL].value_counts().items()}

    df, dropped = drop_hard_columns(df)
    rep["dropped_columns"] = dropped
    feats = feature_columns(df)
    rep["n_features_after_drop"] = len(feats)

    # 1) giá trị không hợp lệ: inf/NaN và giá trị vượt ngưỡng vật lý
    max_abs = float(cfg_cleaning.get("max_abs_value", 1.0e15))
    range_policy = cfg_cleaning.get("out_of_range", "drop")
    bad_nf, bad_rg, nf_by_col, rg_by_col = invalid_value_mask(df, feats, max_abs)
    rep["nonfinite_rows"] = int(bad_nf.sum())
    rep["nonfinite_by_col"] = nf_by_col
    rep["out_of_range_rows"] = int(bad_rg.sum())
    rep["out_of_range_by_col"] = rg_by_col
    rep["out_of_range_threshold"] = max_abs
    rep["out_of_range_policy"] = range_policy
    if rg_by_col:
        log.warning("Có giá trị |x| > %.0e ở các cột %s (%d dòng): nghi giá trị đệm; chính sách '%s'",
                    max_abs, rg_by_col, int(bad_rg.sum()), range_policy)
    drop_mask = np.zeros(len(df), dtype=bool)
    if cfg_cleaning.get("nonfinite", "drop") == "drop":
        drop_mask |= bad_nf
    if range_policy == "drop":
        drop_mask |= bad_rg
    rep["invalid_rows_removed"] = int(drop_mask.sum())
    rep["invalid_rows_removed_by_attack"] = {
        str(k): int(v) for k, v in df.loc[drop_mask, ATTACK_COL].value_counts().items()}
    if drop_mask.any():
        df = df.loc[~drop_mask].reset_index(drop=True)
    rep["rows_after_invalid"] = int(len(df))

    # 2) gộp trùng lặp theo vector đặc trưng, xử lý xung đột nhãn, giữ trọng số
    policy = cfg_cleaning.get("conflict_policy", "majority")
    if cfg_cleaning.get("drop_duplicates", True):
        df, info = collapse_duplicates(df, feats, policy)
        rep["collapse"] = info
    else:
        df[WEIGHT_COL] = np.int32(1)
        df[MINORITY_COL] = np.int32(0)
        rep["collapse"] = {"policy": "none", "rows_in": int(len(df)), "rows_out": int(len(df))}

    # 3) giảm bộ nhớ và chuẩn hóa kiểu
    df = downcast_numeric(df, feats)
    df[ATTACK_COL] = df[ATTACK_COL].astype("category")
    if FAMILY_COL in df.columns:
        df[FAMILY_COL] = df[FAMILY_COL].astype("category")

    w = df[WEIGHT_COL].to_numpy(dtype="float64")
    y = df[LABEL_COL].to_numpy()
    rep["rows_after"] = int(len(df))
    rep["weighted_rows_after"] = int(w.sum())
    rep["attack_ratio_after_unweighted"] = float(y.mean())
    rep["attack_ratio_after_weighted"] = float((w * y).sum() / w.sum())
    rep["attack_unique_after"] = {str(k): int(v) for k, v in df[ATTACK_COL].value_counts().items()}
    rep["attack_weighted_after"] = {
        str(k): int(v) for k, v in df.groupby(ATTACK_COL, observed=True)[WEIGHT_COL].sum().items()}
    rep["features_final"] = feature_columns(df)
    log.info(
        "Làm sạch: %d -> %d vector duy nhất (không hợp lệ %d, trùng %d, xung đột nhãn %d dòng); "
        "tỷ lệ tấn công: gốc %.2f%% | duy nhất %.2f%% | có trọng số %.2f%%",
        rep["rows_before"], rep["rows_after"], rep["invalid_rows_removed"],
        rep["collapse"].get("duplicate_rows_removed", 0), rep["collapse"].get("conflict_rows", 0),
        100 * rep["attack_ratio_before"], 100 * rep["attack_ratio_after_unweighted"],
        100 * rep["attack_ratio_after_weighted"],
    )
    return df, rep
