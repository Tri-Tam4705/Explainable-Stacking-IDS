"""Độ ổn định của giải thích SHAP gộp.

TreeSHAP (path-dependent) là tất định: cùng mô hình và cùng luồng thì luôn cho cùng φ. Vì vậy "ổn định" của SHAP gộp
được đo ở hai mức:
  1. Tính lại trên đúng các luồng đã lưu: thứ hạng từng luồng phải trùng (τ = 1). Đây là phép kiểm tra, không phải thí nghiệm.
  2. Bootstrap tập luồng được giải thích: thứ hạng TOÀN CỤC (mean |φ| theo nhóm hoặc theo đặc trưng) đổi bao nhiêu khi
     tập luồng thay đổi. Đây là độ ổn định của kết luận "nhóm nào quan trọng nhất" trước sai số lấy mẫu.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import kendalltau


def kendall_tau_rows(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Kendall τ-b theo từng hàng giữa A và B (n × J). NaN nếu một hàng không đổi."""
    A, B = np.asarray(A, dtype="float64"), np.asarray(B, dtype="float64")
    out = np.full(len(A), np.nan)
    for i in range(len(A)):
        if np.ptp(A[i]) > 0 and np.ptp(B[i]) > 0:
            out[i] = kendalltau(A[i], B[i]).statistic
    return out


def bootstrap_rank_stability(A: np.ndarray, n_boot: int, rng: np.random.Generator, top_k: int | None = None) -> dict:
    """Độ ổn định của thứ hạng toàn cục mean |A| (theo cột) khi lấy mẫu lại các hàng có hoàn lại.

    A: ma trận đóng góp (n luồng × J nhóm hoặc đặc trưng). Mỗi lần bootstrap lấy n hàng có hoàn lại, tính mean |A| theo cột
    rồi so thứ hạng với thứ hạng trên toàn bộ mẫu bằng Kendall τ-b; nếu có top_k thì thêm Jaccard của tập k cột đứng đầu.
    """
    A = np.abs(np.asarray(A, dtype="float64"))
    n, J = A.shape
    full = A.mean(axis=0)
    top_full = set(np.argsort(-full, kind="stable")[:top_k].tolist()) if top_k else None
    taus, jacs = [], []
    for _ in range(n_boot):
        imp = A[rng.integers(0, n, size=n)].mean(axis=0)
        t = kendalltau(full, imp).statistic if np.ptp(full) > 0 and np.ptp(imp) > 0 else np.nan
        taus.append(float(t))
        if top_full is not None:
            tb = set(np.argsort(-imp, kind="stable")[:top_k].tolist())
            jacs.append(len(top_full & tb) / len(top_full | tb))
    out = {"kendall_tau_mean": float(np.nanmean(taus)) if n_boot else None,
           "kendall_tau_std": float(np.nanstd(taus)) if n_boot else None,
           "per_bootstrap": taus, "n_columns": int(J)}
    if top_full is not None:
        out["top_k"] = int(top_k)
        out["topk_jaccard_mean"] = float(np.mean(jacs)) if jacs else None
    return out
