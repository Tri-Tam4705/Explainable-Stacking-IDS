"""Giá trị Shapley theo NHÓM đặc trưng với hàm giá trị "trong phân phối", tính chính xác.

Người chơi là G nhóm ngữ nghĩa (G = 6), nên liệt kê được đủ 2^G liên minh. Hàm giá trị của liên minh S (các nhóm
được giữ): thay mọi cột của các nhóm ngoài S bằng giá trị của k luồng lành tính gần nhất, khoảng cách tính trên các
cột còn giữ (cùng cách với phép xóa trong phân phối ở checks.py), rồi lấy trung bình logit z của hệ stacking.
v(toàn bộ) = z(x); v(rỗng) = trung bình z trên một tập luồng lành tính cố định.

Khác SHAP gộp (TreeSHAP theo đặc trưng, hàm giá trị path-dependent hoặc interventional), ở đây "vắng mặt" nghĩa là lấy
giá trị từ lưu lượng lành tính thật có phần còn lại giống luồng đang xét, nên vector sau khi xóa nằm trong phân phối.
Đây là Shapley có điều kiện ước lượng bằng láng giềng gần nhất (Aas và cộng sự, 2021; Frye và cộng sự, 2021), áp dụng
cho nhóm đặc trưng. Cùng một bảng liên minh cho thêm:
  - phép xóa g nhóm, mức tối đa (oracle: chọn g nhóm tốt nhất cho từng luồng) và kỳ vọng chính xác khi xóa ngẫu nhiên;
  - tương quan trung thực: tương quan giữa tổng đóng góp của một tập nhóm và mức giảm z khi xóa tập đó;
  - kiểm tra chéo: hồi quy tuyến tính có trọng số kernel Shapley trên đủ 2^G liên minh cho lại đúng giá trị Shapley
    (KernelSHAP, Lundberg và Lee, 2017).
Phần cuối có bản theo đặc trưng (d = 25) cho KernelSHAP với phép xóa interventional, dùng liên minh lấy mẫu.
Các hàm ở đây chỉ tính toán, không đọc ghi file.
"""
from __future__ import annotations

import itertools
from math import comb, factorial

import numpy as np


# ── bảng liên minh ───────────────────────────────────────────────────────────
def all_masks(G: int) -> np.ndarray:
    """Mọi liên minh của G người chơi, mảng 2^G × G (1 = giữ), thứ tự của itertools.product."""
    return np.array(list(itertools.product((0, 1), repeat=G)), dtype=np.int8)


def _index(masks: np.ndarray) -> dict:
    return {tuple(int(v) for v in m): i for i, m in enumerate(masks)}


def coalition_table(X, z_fn, imputer, group_cols, k: int = 25, n_empty: int = 200, seed: int = 0,
                    block: int = 256) -> dict:
    """Giá trị của mọi liên minh nhóm cho từng luồng.

    X: n × d. z_fn: mảng m × d -> logit (m,). imputer: NeighborImputer trên luồng lành tính của train.
    Trả về masks (2^G × G), Z và P (n × 2^G): logit trung bình và xác suất trung bình (trung bình σ(z) trên láng giềng).
    """
    X = np.asarray(X, dtype="float64")
    n, d = X.shape
    G = len(group_cols)
    masks = all_masks(G)
    T = imputer.transform(X)
    Z, P = np.zeros((n, len(masks))), np.zeros((n, len(masks)))
    empty_nb = np.random.default_rng(seed).choice(len(imputer.ref), min(n_empty, len(imputer.ref)), replace=False)
    z_empty = z_fn(imputer.ref[empty_nb])
    for m, mask in enumerate(masks):
        keep = np.concatenate([np.asarray(group_cols[g], dtype=int) for g in range(G) if mask[g]]) if mask.any() \
            else np.array([], dtype=int)
        drop = np.setdiff1d(np.arange(d), keep)
        if len(drop) == 0:
            zz = z_fn(X)
            Z[:, m], P[:, m] = zz, 1.0 / (1.0 + np.exp(-zz))
            continue
        if len(keep) == 0:
            Z[:, m], P[:, m] = z_empty.mean(), (1.0 / (1.0 + np.exp(-z_empty))).mean()
            continue
        Tk = imputer.T[:, keep]
        sq = (Tk ** 2).sum(1)
        kk = min(k, len(imputer.ref))
        for s in range(0, n, block):
            blk = T[s:s + block][:, keep]
            dist = (blk ** 2).sum(1)[:, None] - 2.0 * blk @ Tk.T + sq[None, :]
            nb = np.argpartition(dist, kk - 1, axis=1)[:, :kk]
            cand = np.repeat(X[s:s + block][:, None, :], kk, axis=1)
            cand[:, :, drop] = imputer.ref[nb][:, :, drop]
            zz = z_fn(cand.reshape(-1, d)).reshape(len(blk), kk)
            Z[s:s + block, m], P[s:s + block, m] = zz.mean(1), (1.0 / (1.0 + np.exp(-zz))).mean(1)
    return {"masks": masks, "Z": Z, "P": P}


# ── giá trị Shapley chính xác và kiểm tra chéo bằng kernel Shapley ─────────
def exact_shapley(Z: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Giá trị Shapley chính xác (n × G) từ bảng giá trị đủ 2^G liên minh."""
    G = masks.shape[1]
    idx = _index(masks)
    phi = np.zeros((len(Z), G))
    for j in range(G):
        for mask in masks:
            if mask[j]:
                continue
            s = int(mask.sum())
            w = factorial(s) * factorial(G - s - 1) / factorial(G)
            m1 = mask.copy()
            m1[j] = 1
            phi[:, j] += w * (Z[:, idx[tuple(int(v) for v in m1)]] - Z[:, idx[tuple(int(v) for v in mask)]])
    return phi


def shapley_kernel(masks: np.ndarray, big: float = 1e6) -> np.ndarray:
    """Kernel Shapley (Lundberg và Lee, 2017); liên minh rỗng và đầy đủ nhận trọng số rất lớn (ràng buộc)."""
    G = masks.shape[1]
    s = masks.sum(1)
    return np.array([big if si in (0, G) else (G - 1) / (comb(G, int(si)) * si * (G - si)) for si in s])


def kernel_attribution(Z: np.ndarray, masks: np.ndarray, weights: np.ndarray, ridge: float = 0.0) -> np.ndarray:
    """Hồi quy tuyến tính có trọng số z ~ b + Σ βⱼ·maskⱼ trên mọi liên minh, cho từng luồng; trả về β (n × G).

    ridge: phạt L2 lên β (không phạt hệ số chặn).
    """
    A = np.c_[np.ones(len(masks)), masks.astype("float64")]
    W = np.asarray(weights, dtype="float64")
    reg = ridge * np.eye(A.shape[1])
    reg[0, 0] = 0.0
    beta = np.linalg.solve(A.T @ (W[:, None] * A) + reg, A.T @ (W[:, None] * np.asarray(Z, dtype="float64").T))
    return beta[1:].T


# ── độ trung thực: phép xóa nhóm và tương quan ──────────────────────────────
def deletion_eval(Z: np.ndarray, P: np.ndarray, masks: np.ndarray, attrs: dict, threshold: float, gs=(1, 2, 3)) -> dict:
    """Xóa g nhóm có đóng góp lớn nhất của từng luồng, đọc kết quả từ bảng liên minh (không chạy mô hình lại).

    Với mỗi g: oracle (chọn g nhóm làm mất cảnh báo nếu có, và làm z giảm nhiều nhất), random (kỳ vọng chính xác trên mọi
    tập g nhóm), và từng phương pháp: flip_rate (tỷ lệ mất cảnh báo), mean_dz (mức giảm z trung bình),
    evidence_share_median (trung vị của mức giảm z chia cho z(x) − v(rỗng)), share_of_oracle_flip.
    """
    G = masks.shape[1]
    idx = _index(masks)
    full, empty = idx[(1,) * G], idx[(0,) * G]
    z0, p0 = Z[:, full], P[:, full]
    alert = p0 >= threshold
    total = np.maximum(z0 - Z[:, empty], 1e-12)
    rows = np.arange(len(Z))
    out = {"n_flows": int(len(Z)), "alert_share": float(alert.mean())}
    for g in gs:
        subsets = list(itertools.combinations(range(G), g))
        cols = [idx[tuple(0 if j in R else 1 for j in range(G))] for R in subsets]
        flips = alert[:, None] & (P[:, cols] < threshold)
        dz = z0[:, None] - Z[:, cols]
        oracle = {"flip_rate": float(flips.any(1).mean()), "mean_dz": float(dz.max(1).mean())}
        res = {"oracle": oracle, "random": {"flip_rate": float(flips.mean()), "mean_dz": float(dz.mean())}}
        for name, A in attrs.items():
            top = np.argsort(-np.asarray(A, dtype="float64"), axis=1, kind="stable")[:, :g]
            pick = np.array([subsets.index(tuple(sorted(int(v) for v in t))) for t in top])
            fr, mdz = float(flips[rows, pick].mean()), float(dz[rows, pick].mean())
            res[name] = {"flip_rate": fr, "mean_dz": mdz,
                         "evidence_share_median": float(np.median(dz[rows, pick] / total)),
                         "share_of_oracle_flip": fr / oracle["flip_rate"] if oracle["flip_rate"] > 0 else None,
                         "share_of_oracle_dz": mdz / oracle["mean_dz"] if oracle["mean_dz"] > 0 else None}
        out[str(g)] = res
    return out


def faithfulness_correlation(Z: np.ndarray, masks: np.ndarray, attrs: dict) -> dict:
    """Tương quan trung thực (Bhatt và cộng sự, 2020) đủ mọi tập nhóm R (khác rỗng, khác toàn bộ):
    tương quan Pearson theo từng luồng giữa Σ_{j∈R} φⱼ và z(x) − v(N∖R). Trả về trung bình và trung vị theo luồng."""
    G = masks.shape[1]
    idx = _index(masks)
    full = idx[(1,) * G]
    subsets = [R for g in range(1, G) for R in itertools.combinations(range(G), g)]
    cols = [idx[tuple(0 if j in R else 1 for j in range(G))] for R in subsets]
    D = Z[:, full][:, None] - Z[:, cols]
    Dc = D - D.mean(1, keepdims=True)
    out = {}
    for name, A in attrs.items():
        A = np.asarray(A, dtype="float64")
        S = np.stack([A[:, list(R)].sum(1) for R in subsets], axis=1)
        Sc = S - S.mean(1, keepdims=True)
        den = np.sqrt((Sc ** 2).sum(1) * (Dc ** 2).sum(1))
        r = np.where(den > 0, (Sc * Dc).sum(1) / np.where(den > 0, den, 1.0), np.nan)
        out[name] = {"mean": float(np.nanmean(r)), "median": float(np.nanmedian(r))}
    return out


# ── bản theo đặc trưng: KernelSHAP với phép xóa interventional ─────────────
def sample_masks(d: int, n_pairs: int, rng: np.random.Generator) -> np.ndarray:
    """Liên minh lấy mẫu cho biểu diễn nhị phân: số đặc trưng bị xóa đều trên 1..d−1, chọn đều trong cỡ đó.
    Mỗi mẫu đi kèm phần bù của nó (lấy mẫu ghép cặp) để giảm phương sai. Trả về (2·n_pairs) × d, 1 = giữ."""
    out = np.ones((2 * n_pairs, d), dtype=np.int8)
    for i in range(n_pairs):
        r = int(rng.integers(1, d))
        drop = rng.choice(d, r, replace=False)
        out[2 * i, drop] = 0
        out[2 * i + 1] = 1 - out[2 * i]
    return out


def interventional_values(x, masks: np.ndarray, background, z_fn) -> np.ndarray:
    """v(S) = trung bình trên tập nền của z(x_S, nền_S̄), cho từng liên minh trong masks."""
    x = np.asarray(x, dtype="float64")
    B = np.asarray(background, dtype="float64")
    m = masks.astype(bool)
    rows = np.where(m[:, None, :], x[None, None, :], B[None, :, :])  # M × nB × d
    return z_fn(rows.reshape(-1, len(x))).reshape(len(masks), len(B)).mean(1)


def fit_kernel_shap(masks: np.ndarray, values: np.ndarray, v_full: float, v_empty: float) -> np.ndarray:
    """KernelSHAP với liên minh lấy mẫu theo sample_masks (cỡ đều): trọng số quan trọng tỷ lệ 1/(s(d−s)) và ràng buộc
    Σφ = v_full − v_empty (khử biến cuối)."""
    M = masks.astype("float64")
    d = M.shape[1]
    s = M.sum(1)
    ok = (s > 0) & (s < d)
    M, y = M[ok], np.asarray(values, dtype="float64")[ok] - v_empty
    w = 1.0 / (s[ok] * (d - s[ok]))
    total = v_full - v_empty
    A = M[:, :-1] - M[:, [-1]]
    yy = y - M[:, -1] * total
    phi_head = np.linalg.solve(A.T @ (w[:, None] * A), A.T @ (w * yy))
    return np.r_[phi_head, total - phi_head.sum()]
