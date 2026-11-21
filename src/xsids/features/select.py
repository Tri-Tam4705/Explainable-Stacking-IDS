"""Chọn đặc trưng: sàng theo từng dataset (chỉ train), lấy phần chung, gộp đặc trưng tương quan."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import roc_auc_score

from xsids.features.engineer import GROUP_ORDER


def weighted_auc(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """AUC có trọng số, không phân biệt chiều (>= 0.5)."""
    if len(np.unique(y)) < 2 or np.ptp(x) == 0:
        return 0.5
    a = roc_auc_score(y, x, sample_weight=w)
    return float(max(a, 1.0 - a))


def screen_dataset(X: pd.DataFrame, y: np.ndarray, w: np.ndarray, cfg: dict) -> dict:
    """Mỗi đặc trưng: tỷ lệ 0 có trọng số, AUC có/không trọng số, và các lý do bị loại."""
    out = {}
    wsum = float(w.sum())
    for c in X.columns:
        x = X[c].to_numpy(dtype="float64")
        zero = float((w * (x == 0)).sum() / wsum)
        constant = bool(np.ptp(x) == 0)
        auc_w = weighted_auc(x, y, w)
        auc_u = weighted_auc(x, y, np.ones_like(w))
        reasons = []
        if constant:
            reasons.append("hằng")
        if zero >= cfg["sparse_zero_threshold"]:
            reasons.append(f"thưa (>= {100*cfg['sparse_zero_threshold']:.0f}% giá trị 0)")
        if auc_w >= cfg["shortcut_auc_threshold"]:
            reasons.append(f"lối tắt (AUC >= {cfg['shortcut_auc_threshold']})")
        out[c] = {"zero_frac_w": round(zero, 4), "auc_w": round(auc_w, 4), "auc_unw": round(auc_u, 4),
                  "reasons": reasons}
    return out


def cluster_within_groups(pooled: pd.DataFrame, groups: dict[str, str], auc_mean: dict[str, float],
                          threshold: float):
    """Gộp đặc trưng |Spearman| >= threshold TRONG CÙNG NHÓM; đại diện là đặc trưng có AUC trung bình cao nhất."""
    kept, clusters = [], []
    for grp in GROUP_ORDER:
        cols = [c for c in pooled.columns if groups[c] == grp]
        if not cols:
            continue
        if len(cols) == 1:
            kept += cols
            clusters.append({"group": grp, "representative": cols[0], "members": cols})
            continue
        corr = pooled[cols].rank().corr().abs().fillna(0.0).to_numpy()
        dist = 1.0 - (corr + corr.T) / 2.0
        np.fill_diagonal(dist, 0.0)
        z = linkage(squareform(dist, checks=False), method="average")
        lab = fcluster(z, t=1.0 - threshold, criterion="distance")
        for k in np.unique(lab):
            members = [cols[i] for i in np.flatnonzero(lab == k)]
            rep = sorted(members, key=lambda m: (-auc_mean.get(m, 0.0), m))[0]
            kept.append(rep)
            clusters.append({"group": grp, "representative": rep, "members": members})
    return kept, clusters


def build_feature_set(per_ds: dict, groups: dict[str, str], cfg: dict, seed: int = 0) -> dict:
    """per_ds: {ten: (X_train, y, w)}. Trả về bộ đặc trưng chung và toàn bộ lý do loại."""
    screens = {d: screen_dataset(X, y, w, cfg) for d, (X, y, w) in per_ds.items()}
    cand = list(next(iter(per_ds.values()))[0].columns)

    dropped: dict[str, list[str]] = {}
    survivors = []
    for c in cand:
        why = [f"{d}: {', '.join(screens[d][c]['reasons'])}" for d in screens if screens[d][c]["reasons"]]
        if why:
            dropped[c] = why
        else:
            survivors.append(c)

    rng = np.random.default_rng(seed)
    n_each = max(1, cfg["sample_rows"] // len(per_ds))
    parts = []
    for d, (X, _, _) in per_ds.items():
        take = rng.choice(len(X), size=min(n_each, len(X)), replace=False)
        parts.append(X.iloc[take][survivors])
    pooled = pd.concat(parts, ignore_index=True)

    auc_mean = {c: float(np.mean([screens[d][c]["auc_w"] for d in screens])) for c in survivors}
    final, clusters = cluster_within_groups(pooled, groups, auc_mean, cfg["corr_threshold"])
    for cl in clusters:
        for m in cl["members"]:
            if m != cl["representative"]:
                dropped[m] = [f"trùng lặp với {cl['representative']} (|Spearman| >= {cfg['corr_threshold']})"]
    return {"features": final, "groups": {c: groups[c] for c in final}, "dropped": dropped,
            "clusters": clusters, "screening": screens}
