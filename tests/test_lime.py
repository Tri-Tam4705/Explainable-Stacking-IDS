import numpy as np
import pandas as pd
import pytest

from xsids.explain.lime_stack import (
    binary_columns,
    deletion_test,
    global_importance,
    rank_corr_rows,
    stability_metrics,
    stratified_pick,
    summarize,
    topk_jaccard,
    topk_sign_agreement,
)


def test_stratified_pick_matches_step08_logic():
    """Phải trùng từng phần tử với đoạn lấy mẫu inline trong scripts/08_explain.py."""
    rng0 = np.random.default_rng(7)
    att = rng0.choice(np.array(["DoS", "Exploits", "Fuzzers", "Backdoor"]), size=3000, p=[0.5, 0.3, 0.15, 0.05])
    flagged = np.flatnonzero(rng0.random(3000) < 0.6)

    rng = np.random.default_rng(42)  # nguyên văn logic của 08_explain.py
    names = np.unique(att[flagged])
    per = max(1, 200 // len(names))
    expected = np.sort(np.concatenate([rng.choice(flagged[att[flagged] == a],
                                                  size=min(per, int((att[flagged] == a).sum())), replace=False)
                                       for a in names]))
    got = stratified_pick(flagged, att, 200, np.random.default_rng(42))
    assert np.array_equal(got, expected)


def test_stratified_pick_is_subset_and_covers_small_classes():
    att = np.array(["A"] * 100 + ["B"] * 3)
    got = stratified_pick(np.arange(103), att, 20, np.random.default_rng(0))
    assert set(got) <= set(range(103)) and len(got) == len(set(got))
    assert (att[got] == "B").sum() == 3  # lớp nhỏ được lấy hết, không lỗi khi size > số có


def test_rank_corr_rows_extremes_and_constant_rows():
    A = np.random.default_rng(0).normal(size=(5, 8))
    assert np.allclose(rank_corr_rows(A, A), 1.0)
    assert np.allclose(rank_corr_rows(A, -A), -1.0)
    const = np.ones((1, 8))
    assert np.isnan(rank_corr_rows(const, A[:1])[0])


def test_topk_jaccard_and_sign_agreement():
    a = np.array([[5.0, -4.0, 3.0, 0.1, 0.0]])
    assert topk_jaccard(a, a, 3)[0] == 1.0
    b = np.array([[0.0, 0.1, 3.0, -4.0, 5.0]])           # top-3 của a: {0,1,2}; của b: {4,3,2}
    assert abs(topk_jaccard(a, b, 3)[0] - 1 / 5) < 1e-12  # giao {2}, hợp 5 phần tử
    assert topk_sign_agreement(a, a, 3)[0] == 1.0
    flipped = np.array([[-5.0, 4.0, -3.0, 0.1, 0.0]])
    assert topk_sign_agreement(a, flipped, 3)[0] == 0.0


def test_global_importance_sums_to_one_and_is_scale_free():
    A = np.random.default_rng(1).normal(size=(50, 6))
    g = global_importance(A)
    assert abs(g.sum() - 1) < 1e-12 and np.allclose(g, global_importance(100 * A))


def test_summarize_ignores_nan():
    s = summarize([1.0, 2.0, np.nan, 3.0])
    assert s["n"] == 3 and abs(s["mean"] - 2.0) < 1e-12
    assert summarize([np.nan])["mean"] is None


def test_deletion_test_ranks_true_driver_above_random():
    rng = np.random.default_rng(2)
    n, d = 400, 6
    X = rng.normal(size=(n, d))
    X[:, 0] += 3.0                                   # đặc trưng 0 là nguyên nhân duy nhất của điểm tấn công
    score = lambda A: 1 / (1 + np.exp(-(2.0 * A[:, 0] - 3.0)))  # noqa: E731
    good = np.zeros((n, d)); good[:, 0] = 1.0        # giải thích đúng
    bad = np.zeros((n, d)); bad[:, 5] = 1.0          # giải thích sai
    thr = 0.5
    keep = score(X) >= thr
    out = deletion_test(score, X[keep], {"good": good[keep], "bad": bad[keep]}, np.zeros(d), [1], thr,
                        np.random.default_rng(0))["1"]
    assert out["good"]["mean_drop"] > out["random"]["mean_drop"] > out["bad"]["mean_drop"] - 1e-9
    assert out["good"]["flip_rate"] > 0.9 and out["bad"]["flip_rate"] == 0.0


def test_deletion_skips_features_already_at_baseline():
    """Đặc trưng có giá trị bằng giá trị nền không được chiếm chỗ top-k (xóa nó là phép không làm gì)."""
    n, d = 200, 4
    X = np.zeros((n, d)); X[:, 1] = 5.0                # đặc trưng 0 bằng nền (0), đặc trưng 1 mới là nguyên nhân
    score = lambda A: 1 / (1 + np.exp(-(A[:, 1] - 2.0)))  # noqa: E731
    attr = np.zeros((n, d)); attr[:, 0] = 10.0; attr[:, 1] = 1.0   # giải thích xếp đặc trưng 0 trước
    kw = dict(baseline=np.zeros(d), ks=[1], threshold=0.5, rng=np.random.default_rng(0))
    filt = deletion_test(score, X, {"m": attr}, eligible_only=True, **kw)
    naive = deletion_test(score, X, {"m": attr}, eligible_only=False, **kw)
    assert filt["1"]["m"]["flip_rate"] == 1.0 and naive["1"]["m"]["flip_rate"] == 0.0
    assert filt["n_eligible_mean"] == 1.0


def test_stability_identical_runs_is_perfect_and_noise_lowers_it():
    base = np.random.default_rng(3).normal(size=(20, 10))
    same = stability_metrics(np.stack([base, base, base]), 3)
    assert same["spearman"]["mean"] == pytest.approx(1.0) and same["topk_jaccard"]["mean"] == pytest.approx(1.0)
    assert same["top1_same_rate"] == 1.0
    noisy = np.stack([base + np.random.default_rng(s).normal(scale=3.0, size=base.shape) for s in range(3)])
    assert stability_metrics(noisy, 3)["spearman"]["mean"] < 0.9


def test_binary_columns_detects_flags_only():
    X = pd.DataFrame({"flag": [0.0, 1.0, 1.0, 0.0], "const0": [0.0] * 4, "cont": [0.1, 0.5, 2.0, 7.0],
                      "two_vals": [2.0, 3.0, 2.0, 3.0]})
    assert binary_columns(X) == [0, 1]


@pytest.fixture(scope="module")
def toy_stack():
    pytest.importorskip("lime")
    pytest.importorskip("xgboost")
    pytest.importorskip("lightgbm")
    from xsids.models.zoo import BASE_NAMES, get_params, load_models_cfg, make_model

    rng = np.random.default_rng(5)
    X = pd.DataFrame(rng.normal(size=(3000, 6)), columns=list("abcdef"))
    X["flag"] = (rng.random(3000) < 0.3).astype("float64")
    y = ((2.0 * X["a"] + 1.0 * X["b"] + rng.normal(0, 0.3, 3000)) > 1.0).astype(int).to_numpy()
    params = get_params(load_models_cfg(), fast=True)
    for k in params:
        params[k]["n_estimators"] = 40
    models = {n: make_model(n, params[n], 0, 1).fit(X, y) for n in BASE_NAMES}
    return models, X, y, np.array([0.5, 0.5, 0.5, 0.5]), -1.0


@pytest.mark.parametrize("target", ["proba", "logit"])
def test_lime_on_stack_recovers_true_driver_with_correct_sign(toy_stack, target):
    """Bảo vệ lỗi đã gặp: ở chế độ hồi quy LIME để trọng số thật ở khóa 1, đọc khóa 0 sẽ đảo dấu toàn bộ."""
    from xsids.explain.lime_stack import build_explainer, lime_weights, make_lime_predict_fn, make_proba_fn

    models, X, y, w, b = toy_stack
    feats = list(X.columns)
    p = make_proba_fn(models, w, b, feats)(X.to_numpy())
    rows = X.to_numpy()[np.argsort(-p)[:12]]               # 12 điểm có xác suất tấn công cao nhất
    explainer = build_explainer(X.iloc[:2000], target, seed=0)
    res = lime_weights(explainer, make_lime_predict_fn(models, w, b, feats, target), rows, target, num_samples=1500)
    W = res["weights"]
    assert W.shape == (12, 7) and np.isfinite(W).all()
    ia = feats.index("a")
    assert (W[:, ia] > 0).mean() >= 0.9                   # 'a' đẩy dự đoán về tấn công
    assert (np.argmax(np.abs(W), axis=1) == ia).mean() >= 0.6
    assert np.isfinite(res["r2"]).all() and np.isfinite(res["local_pred"]).all()


def test_lime_results_are_reproducible_for_same_seed(toy_stack):
    from xsids.explain.lime_stack import build_explainer, lime_weights, make_lime_predict_fn

    models, X, y, w, b = toy_stack
    feats = list(X.columns)
    fn = make_lime_predict_fn(models, w, b, feats, "proba")
    rows = X.to_numpy()[:3]
    r1 = lime_weights(build_explainer(X.iloc[:1000], "proba", seed=11), fn, rows, "proba", 800)["weights"]
    r2 = lime_weights(build_explainer(X.iloc[:1000], "proba", seed=11), fn, rows, "proba", 800)["weights"]
    r3 = lime_weights(build_explainer(X.iloc[:1000], "proba", seed=12), fn, rows, "proba", 800)["weights"]
    assert np.allclose(r1, r2) and not np.allclose(r1, r3)


def test_recover_sample_idx_roundtrip_and_tolerates_tiny_float_noise():
    from xsids.explain.lime_stack import recover_sample_idx
    rng = np.random.default_rng(3)
    z = rng.normal(size=500)
    att = rng.choice(np.array(["DoS", "Exploits", "Fuzzers"]), size=500)
    pick = np.sort(rng.choice(500, size=60, replace=False))
    got = recover_sample_idx(z + 1e-12 * rng.normal(size=500), att, z[pick], att[pick])
    assert got is not None and np.array_equal(np.sort(got), pick)


def test_recover_sample_idx_returns_none_when_not_found():
    from xsids.explain.lime_stack import recover_sample_idx
    z = np.arange(10, dtype=float)
    att = np.array(["A"] * 10)
    assert recover_sample_idx(z, att, np.array([3.5]), np.array(["A"])) is None  # không có dòng nào gần 3,5
    assert recover_sample_idx(z, att, np.array([3.0, 3.0]), np.array(["A", "A"])) is None  # hai dòng ghép vào một chỉ số


def test_build_explainer_supports_no_discretization():
    from xsids.explain.lime_stack import build_explainer
    X = pd.DataFrame(np.random.default_rng(0).normal(size=(200, 4)), columns=list("abcd"))
    X["flag"] = (X["a"] > 0).astype(float)
    ex = build_explainer(X, target="logit", discretizer="none", seed=0)
    assert ex.discretizer is None
    with pytest.raises(ValueError):
        build_explainer(X, discretizer="khong-co")


def test_lime_none_discretization_returns_contributions_not_slopes():
    """Mô hình tuyến tính f(x) = a·x: đóng góp đúng phải là a_j·(x_j − mean_j), không phải độ dốc a_j·scale_j."""
    from xsids.explain.lime_stack import build_explainer, lime_weights
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(loc=[5.0, -2.0, 0.0], scale=[10.0, 0.1, 3.0], size=(3000, 3)), columns=list("abc"))
    a = np.array([0.3, 40.0, -1.0])
    fn = lambda A: np.asarray(A, dtype="float64") @ a
    ex = build_explainer(X, target="logit", discretizer="none", seed=0)
    row = X.iloc[[7]].to_numpy()
    W = lime_weights(ex, fn, row, target="logit", num_samples=4000)["weights"][0]
    expected = a * (row[0] - X.mean().to_numpy())
    assert np.allclose(W, expected, rtol=0.05, atol=0.05)
