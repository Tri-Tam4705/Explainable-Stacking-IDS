import numpy as np

from xsids.explain.checks import (
    NeighborImputer,
    compare,
    kendall,
    knn_deletion_test,
    knn_group_deletion_test,
    pairwise,
    perturb,
    stratum_weights,
)


def test_compare_identical_and_reversed():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(30, 8))
    same = compare(A, A, 3)
    assert same["spearman_flow_mean"] > 0.999 and same["topk_jaccard_mean"] == 1.0 and same["top1_same"] == 1.0
    assert same["global_kendall"] > 0.999 and same["global_topk_overlap"] == 3
    other = compare(A, rng.normal(size=(30, 8)), 3)
    assert other["spearman_flow_mean"] < 0.5


def test_pairwise_reports_mean_and_min_over_pairs():
    rng = np.random.default_rng(1)
    base = rng.normal(size=(20, 6))
    mats = [base, base + rng.normal(0, 0.01, base.shape), base + rng.normal(0, 0.01, base.shape)]
    r = pairwise(mats, 2)
    assert r["n_pairs"] == 3 and r["spearman_flow_mean_min"] <= r["spearman_flow_mean_mean"] <= 1.0


def test_kendall_and_stratum_weights_restore_population_shares():
    assert abs(kendall([1, 2, 3], [1, 2, 3]) - 1) < 1e-12 and abs(kendall([1, 2, 3], [3, 2, 1]) + 1) < 1e-12
    strata = np.array(["a", "a", "b", "b"])            # mẫu lấy đều 2 + 2
    w = stratum_weights(strata, np.ones(4), {"a": 90, "b": 10})
    assert np.allclose(w, [45, 45, 5, 5])               # tầng a chiếm 90% dòng cảnh báo thật


def test_perturb_keeps_zero_sign_and_fixed_columns():
    X = np.array([[0.0, 10.0, 1.0], [5.0, 0.0, 0.0]])
    Xp = perturb(X, 0.05, np.random.default_rng(0), fixed_cols=[2])
    assert (Xp[X == 0] == 0).all() and np.array_equal(Xp[:, 2], X[:, 2])
    assert np.all(np.abs(Xp[X != 0] / X[X != 0] - 1) <= 0.05 + 1e-12)


def test_neighbor_imputer_finds_nearest_on_kept_columns():
    ref = np.array([[0.0, 0.0, 100.0], [10.0, 10.0, 0.0], [0.1, 0.0, 0.0]])
    imp = NeighborImputer(ref)
    # trên hai cột đầu, gần hàng 0 và hàng 2; cột thứ ba bị bỏ qua
    assert set(imp.nearest(np.array([0.0, 0.0, 50.0]), [0, 1], k=2)) == {0, 2}
    assert imp.nn_distance(ref).max() < 1e-6


def test_knn_deletion_ranks_true_attribution_above_random():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, size=(500, 4))                # luồng "lành tính": cột 0 quanh 0
    X = np.column_stack([np.full(40, 6.0), rng.normal(0, 1, (40, 3))])   # cột 0 cao bất thường
    score = lambda A: 1 / (1 + np.exp(-(np.asarray(A)[:, 0] - 3)))      # noqa: E731  điểm chỉ phụ thuộc cột 0
    truth = np.tile([1.0, 0, 0, 0], (40, 1))
    out = knn_deletion_test(score, X, {"truth": truth}, NeighborImputer(ref), np.zeros(4), [1], 0.5,
                            np.random.default_rng(1), n_random=3, ood_ref=NeighborImputer(ref))
    assert out["1"]["truth"]["flip_rate"] == 1.0 and out["1"]["random"]["flip_rate"] < 0.5
    assert out["ood"]["nn_dist_knn_fill_median"] < out["ood"]["nn_dist_original_median"]
    vs = out["1"]["truth"]["vs_random"]
    assert vs["mean_diff"] > 0 and vs["ci95"][0] > 0 and len(out["1"]["truth"]["drop_ci95"]) == 2


def test_knn_group_deletion_removes_redundant_evidence():
    # Điểm phụ thuộc max(cột 0, cột 1): hai cột cùng nhóm mang cùng thông tin, xóa một cột không đủ.
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, size=(500, 6))
    X = np.column_stack([np.full(60, 6.0), np.full(60, 6.0), rng.normal(0, 1, (60, 4))])
    score = lambda A: 1 / (1 + np.exp(-(np.maximum(np.asarray(A)[:, 0], np.asarray(A)[:, 1]) - 3)))  # noqa: E731
    groups = [[0, 1], [2, 3], [4, 5]]
    truth = np.tile([1.0, 0.0, 0.0], (60, 1))
    out = knn_group_deletion_test(score, X, {"truth": truth}, groups, NeighborImputer(ref), [1], 0.5,
                                  np.random.default_rng(1), n_random=3)
    r = out["1"]
    assert r["truth"]["flip_rate"] == 1.0 and r["truth"]["n_deleted_mean"] == 2.0
    assert r["random_groups"]["flip_rate"] < 0.6 and r["random_features"]["flip_rate"] < 0.3
    assert r["truth"]["vs_random_features"]["ci95"][0] > 0 and r["truth"]["vs_random_groups"]["mean_diff"] > 0
    # xóa từng cột lẻ (k = 1) không làm mất cảnh báo vì cột còn lại vẫn mang thông tin
    single = knn_deletion_test(score, X, {"truth": np.tile([1.0, 0.5, 0, 0, 0, 0], (60, 1))}, NeighborImputer(ref),
                               np.zeros(6), [1], 0.5, np.random.default_rng(1), n_random=2)
    assert single["1"]["truth"]["flip_rate"] == 0.0
