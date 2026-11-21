"""Chia train/val/test có phân tầng theo tên tấn công, và lưu kèm manifest."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from xsids.schema import ATTACK_COL, LABEL_COL, WEIGHT_COL
from xsids.utils import ensure_dir, write_json


def make_strata(attack: pd.Series, min_stratum_size: int = 10) -> np.ndarray:
    """Gộp các lớp quá nhỏ vào một tầng chung để train_test_split không báo lỗi."""
    a = attack.astype(str)
    counts = a.value_counts()
    rare = set(counts[counts < min_stratum_size].index)
    s = a.where(~a.isin(rare), "__RARE__")
    n_rare = int((s == "__RARE__").sum())
    if 0 < n_rare < 3:  # quá ít để tự thành một tầng: nhập vào lớp nhỏ nhất còn lại
        kept = counts[counts >= min_stratum_size]
        s = s.where(s != "__RARE__", kept.idxmin())
    return s.to_numpy()


def weight_balanced_assign(weights: np.ndarray, fracs: list[float], rng: np.random.Generator,
                           heavy_share: float = 0.0005) -> np.ndarray:
    """Gán từng vector vào một tập (train/val/test).

    - Vector "nặng" (trọng số >= heavy_share của tổng khối lượng lớp): xếp từ nặng đến nhẹ và đặt vào
      tập đang thiếu KHỐI LƯỢNG nhất, để vài vector lặp lại khổng lồ không dồn hết vào một tập.
    - Vector còn lại: đặt vào tập đang thiếu SỐ LƯỢNG nhất (thứ tự đã xáo ngẫu nhiên theo seed), nên
      tỷ lệ 70/10/20 về số vector được giữ đúng.
    """
    n, k = len(weights), len(fracs)
    total = float(weights.sum())
    heavy = weights >= max(heavy_share * total, 1.0)
    out = np.empty(n, dtype=np.int8)
    cur_m, cur_c = [0.0] * k, [0] * k

    idx_h = np.flatnonzero(heavy)
    idx_h = idx_h[np.argsort(-weights[idx_h], kind="stable")] if len(idx_h) else idx_h
    mass_h = float(weights[idx_h].sum()) if len(idx_h) else 0.0
    for i in idx_h.tolist():
        best = max(range(k), key=lambda s: fracs[s] * mass_h - cur_m[s])
        out[i] = best
        cur_m[best] += float(weights[i])
        cur_c[best] += 1

    idx_l = rng.permutation(np.flatnonzero(~heavy)).tolist()
    for i in idx_l:
        best, best_d = 0, -1e300
        for s in range(k):
            d = fracs[s] * n - cur_c[s]
            if d > best_d:
                best, best_d = s, d
        out[i] = best
        cur_c[best] += 1
    return out


def three_way_split(
    df: pd.DataFrame,
    test_size: float = 0.20,
    val_size: float = 0.10,
    seed: int = 42,
    min_stratum_size: int = 10,
) -> dict[str, np.ndarray]:
    """Trả về chỉ số vị trí (iloc) của train/val/test, không giao nhau.

    Có cột n_orig: chia cân bằng trọng số theo từng tên tấn công (xem weight_balanced_assign).
    Không có: chia stratified thông thường theo tên tấn công.
    """
    if WEIGHT_COL in df.columns:
        rng = np.random.default_rng(seed)
        fracs = [1.0 - test_size - val_size, val_size, test_size]  # train, val, test
        attack = df[ATTACK_COL].astype(str).to_numpy()
        w = df[WEIGHT_COL].to_numpy(dtype="float64")
        assign = np.empty(len(df), dtype=np.int8)
        for cls in np.unique(attack):
            idx = np.flatnonzero(attack == cls)
            assign[idx] = weight_balanced_assign(w[idx], fracs, rng)
        return {name: np.flatnonzero(assign == k) for k, name in enumerate(["train", "val", "test"])}

    strata = make_strata(df[ATTACK_COL], min_stratum_size)
    idx = np.arange(len(df))
    idx_trv, idx_te = train_test_split(
        idx, test_size=test_size, random_state=seed, stratify=strata
    )
    rel_val = val_size / (1.0 - test_size)
    idx_tr, idx_va = train_test_split(
        idx_trv, test_size=rel_val, random_state=seed, stratify=strata[idx_trv]
    )
    return {"train": np.sort(idx_tr), "val": np.sort(idx_va), "test": np.sort(idx_te)}


def split_summary(df: pd.DataFrame, parts: dict[str, np.ndarray]) -> dict:
    out = {}
    for name, idx in parts.items():
        sub = df.iloc[idx]
        out[name] = {
            "rows": int(len(sub)),
            "attack_ratio": float(sub[LABEL_COL].mean()),
            "by_attack": {str(k): int(v) for k, v in sub[ATTACK_COL].value_counts().items()},
        }
        if WEIGHT_COL in sub.columns:
            w = sub[WEIGHT_COL].to_numpy(dtype="float64")
            out[name]["weighted_rows"] = int(w.sum())
            out[name]["weighted_attack_ratio"] = float((w * sub[LABEL_COL].to_numpy()).sum() / w.sum())
            out[name]["effective_n"] = int(w.sum() ** 2 / (w ** 2).sum())  # cỡ mẫu hiệu dụng Kish
            lab = sub[LABEL_COL].to_numpy()
            for cls, nm in ((0, "benign"), (1, "attack")):
                wc = w[lab == cls]
                out[name][f"effective_n_{nm}"] = int(wc.sum() ** 2 / (wc ** 2).sum()) if len(wc) else 0
            out[name]["max_weight_share"] = float(w.max() / w.sum())
    return out


def save_splits(df: pd.DataFrame, parts: dict[str, np.ndarray], out_dir, meta: dict | None = None):
    out_dir = ensure_dir(out_dir)
    for name, idx in parts.items():
        df.iloc[idx].reset_index(drop=True).to_parquet(out_dir / f"{name}.parquet", index=False)
    manifest = {"summary": split_summary(df, parts), **(meta or {})}
    write_json(manifest, out_dir / "manifest.json")
    return manifest
