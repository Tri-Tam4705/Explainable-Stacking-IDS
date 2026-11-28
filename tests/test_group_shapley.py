import numpy as np
import pytest

from xsids.explain.checks import NeighborImputer
from xsids.explain.group_shapley import (
    all_masks,
    coalition_table,
    deletion_eval,
    exact_shapley,
    faithfulness_correlation,
    fit_kernel_shap,
    interventional_values,
    kernel_attribution,
    sample_masks,
    shapley_kernel,
)


def _additive_table(n=7, G=4, seed=0):
    rng = np.random.default_rng(seed)
    a, c = rng.normal(size=(n, G)), rng.normal(size=n)
    masks = all_masks(G)
    return masks, c[:, None] + (masks @ a.T).T, a  # v(S) = c + Σ_{j∈S} aⱼ


def _random_table(n=5, G=4, seed=1):
    masks = all_masks(G)
    return masks, np.random.default_rng(seed).normal(size=(n, len(masks)))


def test_all_masks_lists_every_coalition_once():
    m = all_masks(3)
    assert m.shape == (8, 3) and len({tuple(r) for r in m}) == 8


def test_exact_shapley_recovers_additive_game_and_is_efficient():
    masks, Z, a = _additive_table()
    assert np.allclose(exact_shapley(Z, masks), a)
    masks, Z = _random_table()
    phi = exact_shapley(Z, masks)
    assert np.allclose(phi.sum(1), Z[:, -1] - Z[:, 0])  # hàng cuối là liên minh đầy đủ, hàng đầu là rỗng


def test_exact_shapley_gives_symmetric_players_equal_value():
    G = 3
    masks = all_masks(G)
    v = lambda m: 2.0 * (m[0] and m[1]) + 0.5 * m[2]  # người chơi 0 và 1 đối xứng
    Z = np.array([[v(m) for m in masks]], dtype=float)
    phi = exact_shapley(Z, masks)[0]
    assert np.isclose(phi[0], phi[1]) and np.isclose(phi.sum(), v((1, 1, 1)))


def test_shapley_kernel_regression_over_all_coalitions_equals_exact_shapley():
    masks, Z = _random_table(G=5)
    assert np.allclose(kernel_attribution(Z, masks, shapley_kernel(masks)), exact_shapley(Z, masks), atol=1e-4)


def test_coalition_table_matches_linear_model_with_constant_reference():
    rng = np.random.default_rng(2)
    beta = np.array([1.0, -2.0, 0.5, 3.0, 1.5])
    X = rng.normal(size=(6, 5))
    r = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    imp = NeighborImputer(np.repeat(r[None, :], 50, axis=0))
    groups = [[0, 1], [2], [3, 4]]
    tab = coalition_table(X, lambda A: A @ beta, imp, groups, k=5, n_empty=20)
    masks, Z = tab["masks"], tab["Z"]
    assert np.allclose(Z[:, -1], X @ beta) and np.allclose(Z[:, 0], r @ beta)
    phi = exact_shapley(Z, masks)
    expect = np.stack([((X - r) * beta)[:, g].sum(1) for g in groups], axis=1)
    assert np.allclose(phi, expect)
    assert np.allclose(tab["P"][:, -1], 1 / (1 + np.exp(-(X @ beta))))


def test_deletion_eval_oracle_bounds_methods_and_random_is_exact_expectation():
    masks, Z = _random_table(n=40, G=4, seed=3)
    P = 1 / (1 + np.exp(-Z))
    Z[:, -1], P[:, -1] = 3.0, 0.95  # mọi luồng đều đang bị cảnh báo
    attrs = {"a": np.random.default_rng(4).normal(size=(40, 4)), "exact": exact_shapley(Z, masks)}
    res = deletion_eval(Z, P, masks, attrs, threshold=0.5, gs=(1, 2))
    for g in ("1", "2"):
        for name in attrs:
            assert res[g][name]["flip_rate"] <= res[g]["oracle"]["flip_rate"] + 1e-12
            assert res[g][name]["mean_dz"] <= res[g]["oracle"]["mean_dz"] + 1e-12
    idx = {tuple(m): i for i, m in enumerate(masks.tolist())}
    cols = [idx[tuple(0 if j == R else 1 for j in range(4))] for R in range(4)]
    assert np.isclose(res["1"]["random"]["mean_dz"], (Z[:, -1][:, None] - Z[:, cols]).mean())


def test_additive_game_top_g_by_exact_shapley_reaches_the_oracle():
    masks, Z, a = _additive_table(n=30, G=5, seed=5)
    P = np.full_like(Z, 0.9)
    res = deletion_eval(Z, P, masks, {"exact": exact_shapley(Z, masks)}, threshold=0.5, gs=(1, 2, 3))
    for g in ("1", "2", "3"):
        assert np.isclose(res[g]["exact"]["share_of_oracle_dz"], 1.0)


def test_faithfulness_correlation_is_one_for_exact_values_of_an_additive_game():
    masks, Z, a = _additive_table(n=10, G=5, seed=6)
    fc = faithfulness_correlation(Z, masks, {"exact": a, "noise": np.random.default_rng(7).normal(size=a.shape)})
    assert fc["exact"]["mean"] == pytest.approx(1.0) and fc["noise"]["mean"] < 0.9


def test_sample_masks_are_paired_complements_without_trivial_coalitions():
    m = sample_masks(8, 50, np.random.default_rng(8))
    assert m.shape == (100, 8)
    assert np.array_equal(m[0::2] + m[1::2], np.ones((50, 8), dtype=np.int8))
    assert ((m.sum(1) > 0) & (m.sum(1) < 8)).all()


def test_sampled_kernel_shap_is_exact_for_a_linear_model():
    rng = np.random.default_rng(9)
    beta = rng.normal(size=6)
    x, B = rng.normal(size=6), rng.normal(size=(20, 6))
    z_fn = lambda A: A @ beta
    masks = sample_masks(6, 40, rng)
    v = interventional_values(x, masks, B, z_fn)
    expect = beta * (x - B.mean(0))  # SHAP interventional của mô hình tuyến tính
    assert np.allclose(fit_kernel_shap(masks, v, float(z_fn(x[None])[0]), float(z_fn(B).mean())), expect)
