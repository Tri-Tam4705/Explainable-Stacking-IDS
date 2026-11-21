"""Các phép kiểm tra bổ sung cho lời giải thích (scripts/22_explain_checks.py, 23_sanity_randomization.py).

- So hai bộ đóng góp (cùng luồng): tương quan hạng từng luồng, Jaccard top-k, tỷ lệ trùng hạng 1, Kendall τ toàn cục.
- Trọng số khôi phục tỷ lệ của dòng cảnh báo thật khi mẫu được lấy phân tầng theo tên tấn công.
- Nhiễu nhỏ lên đầu vào để đo độ bền cục bộ.
- Phép xóa đặc trưng "trong phân phối": thay đặc trưng bị xóa bằng giá trị của các luồng lành tính gần nhất (tính
  khoảng cách trên các đặc trưng còn giữ), thay cho trung vị lành tính. Vector sau khi xóa vì vậy giống lưu lượng thật
  hơn, và phép thử không dùng chung kiểu can thiệp với SHAP.
Các hàm ở đây chỉ tính toán, không đọc ghi file.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import kendalltau

from xsids.explain.lime_stack import rank_corr_rows, topk_jaccard


# ── so sánh hai bộ đóng góp ──────────────────────────────────────────────────
def kendall(a, b) -> float:
    """Kendall τ giữa hai vector tầm quan trọng (cùng thứ tự đặc trưng)."""
    t = kendalltau(np.asarray(a, dtype="float64"), np.asarray(b, dtype="float64")).statistic
    return float(t) if t == t else float("nan")


def global_importance(A, weights=None) -> np.ndarray:
    """Trung bình |đóng góp| theo cột, có thể có trọng số theo luồng."""
    A = np.abs(np.asarray(A, dtype="float64"))
    return np.average(A, axis=0, weights=weights)


def compare(A, B, k: int = 5, weights=None) -> dict:
    """So hai ma trận đóng góp n × d trên cùng các luồng.

    spearman_flow: tương quan hạng của |đóng góp| trên từng luồng (trung bình, trung vị)
    topk_jaccard : Jaccard của k đặc trưng |đóng góp| lớn nhất trên từng luồng (trung bình)
    top1_same    : tỷ lệ luồng có cùng đặc trưng hạng 1
    global_kendall, global_topk_overlap: so thứ hạng toàn cục mean |đóng góp|
    """
    A, B = np.asarray(A, dtype="float64"), np.asarray(B, dtype="float64")
    rho = rank_corr_rows(np.abs(A), np.abs(B))
    k = min(k, A.shape[1])
    ga, gb = global_importance(A, weights), global_importance(B, weights)
    return {"spearman_flow_mean": float(np.nanmean(rho)), "spearman_flow_median": float(np.nanmedian(rho)),
            "topk_jaccard_mean": float(np.mean(topk_jaccard(A, B, k))),
            "top1_same": float((np.argmax(np.abs(A), axis=1) == np.argmax(np.abs(B), axis=1)).mean()),
            "global_kendall": kendall(ga, gb),
            "global_topk_overlap": int(len(set(np.argsort(-ga)[:k]) & set(np.argsort(-gb)[:k]))), "k": int(k)}


def pairwise(mats: list, k: int = 5, weights=None) -> dict:
    """compare() trên mọi cặp; trả về trung bình và giá trị nhỏ nhất của từng chỉ số."""
    rows = [compare(mats[i], mats[j], k, weights) for i in range(len(mats)) for j in range(i + 1, len(mats))]
    if not rows:
        return {}
    keys = [key for key in rows[0] if key != "k"]
    return {**{f"{key}_mean": float(np.nanmean([r[key] for r in rows])) for key in keys},
            **{f"{key}_min": float(np.nanmin([r[key] for r in rows])) for key in keys}, "n_pairs": len(rows), "k": k}


# ── trọng số khôi phục tỷ lệ ─────────────────────────────────────────────────
def stratum_weights(strata, n_orig, population_counts: dict) -> np.ndarray:
    """Trọng số n_orig × (số vector của tầng trong tổng thể / số vector của tầng trong mẫu).

    Mẫu lấy đều theo tầng (tên tấn công) nên tầng hiếm được đại diện quá mức; trọng số này đưa về tỷ lệ của dòng
    cảnh báo thật. Tầng không có trong population_counts nhận trọng số theo n_orig.
    """
    strata = np.asarray(strata).astype(str)
    w = np.asarray(n_orig, dtype="float64").copy()
    for s in np.unique(strata):
        m = strata == s
        w[m] *= float(population_counts.get(s, m.sum())) / m.sum()
    return w


# ── nhiễu nhỏ lên đầu vào ────────────────────────────────────────────────────
def perturb(X, rel: float, rng: np.random.Generator, fixed_cols=()) -> np.ndarray:
    """Nhân mỗi giá trị với (1 + u), u ~ U(−rel, rel); các cột trong fixed_cols (cờ nhị phân) giữ nguyên.

    Giá trị 0 vẫn là 0, dấu không đổi, nên vector mới vẫn hợp lệ với các đặc trưng không âm.
    """
    X = np.asarray(X, dtype="float64")
    noise = 1.0 + rng.uniform(-rel, rel, size=X.shape)
    noise[:, list(fixed_cols)] = 1.0
    return X * noise


# ── phép xóa đặc trưng trong phân phối ───────────────────────────────────────
def _tf(X) -> np.ndarray:
    X = np.asarray(X, dtype="float64")
    return np.sign(X) * np.log1p(np.abs(X))


class NeighborImputer:
    """Tìm luồng tham chiếu gần nhất trên một tập con đặc trưng (khoảng cách Euclid sau log1p và chuẩn hóa)."""

    def __init__(self, ref):
        self.ref = np.asarray(ref, dtype="float64")
        T = _tf(self.ref)
        self.mu = T.mean(axis=0)
        sd = T.std(axis=0)
        self.sd = np.where(sd > 1e-12, sd, 1.0)
        self.T = (T - self.mu) / self.sd

    def transform(self, X) -> np.ndarray:
        return (_tf(X) - self.mu) / self.sd

    def nearest(self, x, keep_cols, k: int = 5) -> np.ndarray:
        """Chỉ số k luồng tham chiếu gần x nhất, chỉ tính trên keep_cols."""
        t = self.transform(np.asarray(x, dtype="float64")[None, :])[0]
        keep = np.asarray(keep_cols)
        d = ((self.T[:, keep] - t[keep]) ** 2).sum(axis=1)
        k = min(k, len(d))
        return np.argpartition(d, k - 1)[:k]

    def nn_distance(self, X) -> np.ndarray:
        """Khoảng cách tới luồng tham chiếu gần nhất (mọi đặc trưng), đo mức "ngoài phân phối"."""
        T = self.transform(X)
        out = np.empty(len(T))
        for i in range(0, len(T), 256):
            blk = T[i:i + 256]
            d = (blk ** 2).sum(1)[:, None] - 2 * blk @ self.T.T + (self.T ** 2).sum(1)[None, :]
            out[i:i + 256] = np.sqrt(np.maximum(d.min(axis=1), 0))
        return out


def top_positive_cols(A, eligible, k: int) -> np.ndarray:
    """k cột có đóng góp lớn nhất (dương trước) trong các cột hợp lệ của từng hàng."""
    return np.argsort(-np.where(eligible, A, -np.inf), axis=1, kind="stable")[:, :k]


def _knn_candidates(X: np.ndarray, cols_list, imputer: NeighborImputer, n_neighbors: int) -> np.ndarray:
    """Với mỗi hàng i: n_neighbors bản sao của X[i], các cột cols_list[i] lấy từ các luồng tham chiếu gần nhất
    (khoảng cách tính trên các cột còn giữ). Trả về mảng n × K × d."""
    n, d = X.shape
    rows = []
    for i in range(n):
        cols = np.asarray(cols_list[i], dtype=int)
        keep = np.setdiff1d(np.arange(d), cols)
        nb = imputer.nearest(X[i], keep, n_neighbors)
        cand = np.repeat(X[i][None, :], len(nb), axis=0)
        cand[:, cols] = imputer.ref[np.ix_(nb, cols)]
        rows.append(cand)
    return np.stack(rows)


def knn_fill_scores(score_fn, X, cols_list, imputer: NeighborImputer, n_neighbors: int = 5) -> np.ndarray:
    """Xác suất tấn công của từng hàng sau khi xóa cols_list[i] trong phân phối (trung bình trên n_neighbors láng giềng)."""
    X = np.asarray(X, dtype="float64")
    C = _knn_candidates(X, cols_list, imputer, n_neighbors)
    return score_fn(C.reshape(-1, X.shape[1])).reshape(len(X), -1).mean(axis=1)


def boot_mean_ci(x, rng: np.random.Generator, n_boot: int = 2000) -> list[float]:
    """Khoảng tin cậy 95% bootstrap (theo luồng) của trung bình x."""
    x = np.asarray(x, dtype="float64")
    m = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    return [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]


def _paired(drop_a, drop_b, rng, n_boot: int) -> dict:
    """Chênh lệch ghép cặp theo luồng của mức giảm xác suất (a − b) và tỷ lệ luồng a giảm nhiều hơn b."""
    diff = np.asarray(drop_a) - np.asarray(drop_b)
    return {"mean_diff": float(diff.mean()), "ci95": boot_mean_ci(diff, rng, n_boot), "share_a_larger": float((diff > 0).mean())}


def knn_deletion_test(score_fn, X, attributions: dict, imputer: NeighborImputer, baseline, ks, threshold: float,
                      rng: np.random.Generator, n_random: int = 5, n_neighbors: int = 5, ood_ref: NeighborImputer | None = None,
                      ood_k: int | None = None, n_boot: int = 2000, boot_seed: int = 0) -> dict:
    """Phép xóa đặc trưng trong phân phối.

    Với mỗi luồng và mỗi bộ k đặc trưng bị xóa: lấy n_neighbors luồng lành tính gần nhất theo các đặc trưng còn giữ,
    thay các đặc trưng bị xóa bằng giá trị của từng luồng đó, rồi lấy trung bình xác suất tấn công. Cột hợp lệ và thứ tự
    xếp hạng giống deletion_test (cột khác trung vị lành tính, đóng góp dương lớn nhất trước), để hai phép thử so được.
    Trả về, cho mỗi phương pháp và mỗi k: mean_drop, flip_rate, drop_ci95 (bootstrap theo luồng) và vs_random (chênh
    lệch mức giảm so với xóa ngẫu nhiên, ghép cặp theo luồng, kèm khoảng tin cậy 95%). Với ood_ref, thêm khoảng cách tới
    dữ liệu thật của các vector sau khi xóa ood_k đặc trưng (mặc định 5 nếu có, nếu không thì k lớn nhất) theo cách này
    và theo cách thay bằng trung vị.
    """
    X = np.asarray(X, dtype="float64")
    n, d = X.shape
    p0 = score_fn(X)
    eligible = X != np.asarray(baseline)
    boot_rng = np.random.default_rng(boot_seed)  # riêng, để không đổi chuỗi ngẫu nhiên của mốc đối chứng

    def build(cols: np.ndarray) -> np.ndarray:
        return _knn_candidates(X, cols, imputer, n_neighbors)  # n × K × d

    def run(keys: np.ndarray, k: int) -> tuple[np.ndarray, float]:
        C = build(top_positive_cols(keys, eligible, k))
        pk = score_fn(C.reshape(-1, d)).reshape(n, -1).mean(axis=1)
        return p0 - pk, float(((p0 >= threshold) & (pk < threshold)).mean())

    out: dict = {str(k): {} for k in ks}
    drops: dict = {str(k): {} for k in ks}
    for name, A in attributions.items():
        for k in ks:
            dr, fl = run(np.asarray(A, dtype="float64"), k)
            drops[str(k)][name] = dr
            out[str(k)][name] = {"mean_drop": float(dr.mean()), "flip_rate": fl, "drop_ci95": boot_mean_ci(dr, boot_rng, n_boot)}
    for k in ks:
        runs = [run(rng.random((n, d)), k) for _ in range(n_random)]
        dr = np.mean([r[0] for r in runs], axis=0)
        out[str(k)]["random"] = {"mean_drop": float(dr.mean()), "flip_rate": float(np.mean([r[1] for r in runs])),
                                 "drop_ci95": boot_mean_ci(dr, boot_rng, n_boot)}
        for name in attributions:
            out[str(k)][name]["vs_random"] = _paired(drops[str(k)][name], dr, boot_rng, n_boot)
    if ood_ref is not None:
        kmax = ood_k if ood_k is not None else (5 if 5 in list(ks) else max(ks))
        name = next(iter(attributions))
        cols = top_positive_cols(np.asarray(attributions[name], dtype="float64"), eligible, kmax)
        Xm = X.copy()
        Xm[np.arange(n)[:, None], cols] = np.asarray(baseline)[cols]
        Xk = build(cols)[:, 0, :]
        out["ood"] = {"method": name, "k": int(kmax),
                      "nn_dist_original_median": float(np.median(ood_ref.nn_distance(X))),
                      "nn_dist_median_fill_median": float(np.median(ood_ref.nn_distance(Xm))),
                      "nn_dist_knn_fill_median": float(np.median(ood_ref.nn_distance(Xk)))}
    return out


def knn_group_deletion_test(score_fn, X, group_attr: dict, group_cols, imputer: NeighborImputer, gs, threshold: float,
                            rng: np.random.Generator, n_random: int = 5, n_neighbors: int = 5, n_boot: int = 2000,
                            boot_seed: int = 0) -> dict:
    """Phép xóa trong phân phối theo NHÓM đặc trưng.

    Lời giải thích của đồ án trình bày ở mức nhóm, và các đặc trưng trong một nhóm mang thông tin trùng nhau, nên xóa
    vài đặc trưng lẻ để lại phần thông tin đó ở các đặc trưng cùng nhóm. Phép thử này xóa trọn g nhóm có đóng góp lớn
    nhất (dương trước) của từng luồng, thay mọi cột của các nhóm đó bằng giá trị của các luồng lành tính gần nhất.

    `group_attr`: tên phương pháp -> ma trận n × G đóng góp theo nhóm; `group_cols`: danh sách G mảng chỉ số cột.
    Hai mốc đối chứng: xóa g nhóm ngẫu nhiên, và xóa ngẫu nhiên đúng số cột mà phương pháp đầu tiên đã xóa ở từng luồng
    (để phần hơn của phương pháp không đến từ việc xóa nhiều cột hơn). Mỗi phương pháp có vs_random_features và
    vs_random_groups: chênh lệch mức giảm xác suất ghép cặp theo luồng, kèm khoảng tin cậy 95% bootstrap.
    """
    X = np.asarray(X, dtype="float64")
    n, d = X.shape
    p0 = score_fn(X)
    group_cols = [np.asarray(c, dtype=int) for c in group_cols]
    G = len(group_cols)
    boot_rng = np.random.default_rng(boot_seed)  # riêng, để không đổi chuỗi ngẫu nhiên của mốc đối chứng

    def summarize(pk: np.ndarray) -> dict:
        dr = p0 - pk
        return {"mean_drop": float(dr.mean()), "flip_rate": float(((p0 >= threshold) & (pk < threshold)).mean()),
                "drop_ci95": boot_mean_ci(dr, boot_rng, n_boot)}

    def random_runs(make_cols) -> tuple[np.ndarray, float]:
        pks = [knn_fill_scores(score_fn, X, make_cols(), imputer, n_neighbors) for _ in range(n_random)]
        return np.mean([p0 - pk for pk in pks], axis=0), float(np.mean([((p0 >= threshold) & (pk < threshold)).mean() for pk in pks]))

    out: dict = {}
    for g in gs:
        res: dict = {}
        drops: dict = {}
        sizes = None
        for name, A in group_attr.items():
            order = np.argsort(-np.asarray(A, dtype="float64"), axis=1, kind="stable")[:, :g]
            cols = [np.concatenate([group_cols[j] for j in o]) for o in order]
            pk = knn_fill_scores(score_fn, X, cols, imputer, n_neighbors)
            res[name] = {**summarize(pk), "n_deleted_mean": float(np.mean([len(c) for c in cols]))}
            drops[name] = p0 - pk
            if sizes is None:
                sizes = [len(c) for c in cols]
        dr_g, fl_g = random_runs(lambda: [np.concatenate([group_cols[j] for j in rng.permutation(G)[:g]]) for _ in range(n)])
        dr_f, fl_f = random_runs(lambda: [rng.permutation(d)[:s] for s in sizes])
        res["random_groups"] = {"mean_drop": float(dr_g.mean()), "flip_rate": fl_g, "drop_ci95": boot_mean_ci(dr_g, boot_rng, n_boot)}
        res["random_features"] = {"mean_drop": float(dr_f.mean()), "flip_rate": fl_f, "drop_ci95": boot_mean_ci(dr_f, boot_rng, n_boot),
                                  "n_deleted_mean": float(np.mean(sizes))}
        for name in group_attr:
            res[name]["vs_random_features"] = _paired(drops[name], dr_f, boot_rng, n_boot)
            res[name]["vs_random_groups"] = _paired(drops[name], dr_g, boot_rng, n_boot)
        out[str(g)] = res
    return out


def pick_common(explain_idx, attack, n: int, seed: int) -> np.ndarray:
    """Tập luồng chung cho phép so qua seed và phép kiểm tra ngẫu nhiên hóa nhãn (cùng seed thì cùng tập)."""
    from xsids.explain.lime_stack import stratified_pick

    return np.sort(stratified_pick(np.asarray(explain_idx), np.asarray(attack), n, np.random.default_rng(seed + 20)))
