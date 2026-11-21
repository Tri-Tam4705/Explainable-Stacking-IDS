"""Kiểm chứng điểm đồng thuận A(x) như tín hiệu tin cậy của cảnh báo.

Câu hỏi: trong số luồng bị gắn cờ, A(x) thấp có đi kèm cảnh báo sai (FP) nhiều hơn không, và A(x) có mang thông tin
ngoài độ tự tin của mô hình (logit z) không. Các hàm ở đây chỉ tính toán, không đọc ghi file.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def weighted_auc(score, y, w=None) -> float:
    """ROC-AUC của score khi phân biệt y = 1 với y = 0 (có trọng số). NaN nếu chỉ có một lớp."""
    score, y = np.asarray(score, dtype="float64"), np.asarray(y).astype(int)
    keep = ~np.isnan(score)
    score, y = score[keep], y[keep]
    w = None if w is None else np.asarray(w, dtype="float64")[keep]
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, score, sample_weight=w))


def stratified_auc(score, y, strata, w=None) -> float:
    """AUC trung bình trong từng tầng (ví dụ các khoảng tứ phân vị của z), trọng số theo số cặp dương-âm của tầng.

    Nếu A(x) chỉ phản ánh độ tự tin z thì trong mỗi tầng z gần như cố định, AUC của A(x) về gần 0,5.
    """
    score, y, strata = np.asarray(score, dtype="float64"), np.asarray(y).astype(int), np.asarray(strata)
    w = np.ones(len(y)) if w is None else np.asarray(w, dtype="float64")
    num = den = 0.0
    for s in np.unique(strata):
        m = (strata == s) & ~np.isnan(score)
        if len(np.unique(y[m])) < 2:
            continue
        pairs = w[m][y[m] == 1].sum() * w[m][y[m] == 0].sum()
        num += pairs * weighted_auc(score[m], y[m], w[m])
        den += pairs
    return float(num / den) if den > 0 else float("nan")


def quantile_bins(x, q: int = 4) -> np.ndarray:
    """Chỉ số khoảng phân vị 0..q-1 của từng phần tử (bỏ qua NaN khi tính mốc; NaN nhận -1)."""
    x = np.asarray(x, dtype="float64")
    edges = np.nanquantile(x, np.linspace(0, 1, q + 1)[1:-1])
    out = np.searchsorted(edges, x, side="right")
    out[np.isnan(x)] = -1
    return out


def rate_by_bins(bins, y, w=None, q: int = 4) -> list[float]:
    """Tỷ lệ y = 1 (có trọng số) trong từng khoảng 0..q-1."""
    bins, y = np.asarray(bins), np.asarray(y, dtype="float64")
    w = np.ones(len(y)) if w is None else np.asarray(w, dtype="float64")
    out = []
    for k in range(q):
        m = bins == k
        out.append(float((w[m] * y[m]).sum() / w[m].sum()) if w[m].sum() > 0 else float("nan"))
    return out


def bootstrap_ci(fn, n: int, n_boot: int = 500, seed: int = 0, alpha: float = 0.05) -> list[float]:
    """Khoảng tin cậy bootstrap phân vị của fn(idx) khi lấy mẫu lại có hoàn lại n phần tử."""
    rng = np.random.default_rng(seed)
    vals = np.array([fn(rng.integers(0, n, n)) for _ in range(n_boot)], dtype="float64")
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return [float("nan"), float("nan")]
    return [float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))]
