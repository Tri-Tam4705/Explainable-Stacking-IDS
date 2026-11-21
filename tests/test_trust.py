import numpy as np

from xsids.explain.trust import bootstrap_ci, quantile_bins, rate_by_bins, stratified_auc, weighted_auc


def test_weighted_auc_extremes_and_single_class():
    y = np.array([0, 0, 1, 1])
    assert weighted_auc([0.1, 0.2, 0.8, 0.9], y) == 1.0
    assert weighted_auc([0.9, 0.8, 0.2, 0.1], y) == 0.0
    assert np.isnan(weighted_auc([0.1, 0.2], [1, 1]))
    assert weighted_auc([0.1, np.nan, 0.8, 0.9], y) == 1.0          # bỏ NaN


def test_stratified_auc_removes_information_explained_by_the_strata():
    rng = np.random.default_rng(0)
    z = rng.normal(size=4000)
    y = (rng.random(4000) < 1 / (1 + np.exp(3 * z))).astype(int)    # y phụ thuộc z
    a = z + rng.normal(0, 0.01, 4000)                                  # a gần như chỉ là z
    assert weighted_auc(-a, y) > 0.8                                   # nhìn chung a "dự báo" được y
    bins = quantile_bins(z, 20)
    assert abs(stratified_auc(-a, y, bins) - 0.5) < 0.12               # trong cùng khoảng z thì gần ngẫu nhiên


def test_quantile_bins_and_rates():
    x = np.array([1.0, 2, 3, 4, 5, 6, 7, 8, np.nan])
    b = quantile_bins(x, 4)
    assert b[-1] == -1 and sorted(set(b[:-1])) == [0, 1, 2, 3]
    r = rate_by_bins(b, [1, 1, 0, 0, 0, 0, 1, 1, 1], q=4)
    assert r == [1.0, 0.0, 0.0, 1.0]


def test_bootstrap_ci_brackets_the_mean():
    v = np.random.default_rng(1).normal(5, 1, 500)
    lo, hi = bootstrap_ci(lambda i: v[i].mean(), len(v), 300, seed=0)
    assert lo < v.mean() < hi and hi - lo < 0.5
