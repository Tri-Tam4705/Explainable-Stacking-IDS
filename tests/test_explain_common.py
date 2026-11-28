import numpy as np
import pandas as pd

from xsids.explain.common import (
    binary_columns,
    deletion_test,
    explain_subsample,
    global_importance,
    rank_corr_rows,
    recover_sample_idx,
    stratified_pick,
    summarize,
    topk_jaccard,
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


def test_topk_jaccard():
    a = np.array([[5.0, -4.0, 3.0, 0.1, 0.0]])
    assert topk_jaccard(a, a, 3)[0] == 1.0
    b = np.array([[0.0, 0.1, 3.0, -4.0, 5.0]])           # top-3 của a: {0,1,2}; của b: {4,3,2}
    assert abs(topk_jaccard(a, b, 3)[0] - 1 / 5) < 1e-12  # giao {2}, hợp 5 phần tử


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


def test_binary_columns_detects_flags_only():
    X = pd.DataFrame({"flag": [0.0, 1.0, 1.0, 0.0], "const0": [0.0] * 4, "cont": [0.1, 0.5, 2.0, 7.0],
                      "two_vals": [2.0, 3.0, 2.0, 3.0]})
    assert binary_columns(X) == [0, 1]


def test_recover_sample_idx_roundtrip_and_tolerates_tiny_float_noise():
    rng = np.random.default_rng(3)
    z = rng.normal(size=500)
    att = rng.choice(np.array(["DoS", "Exploits", "Fuzzers"]), size=500)
    pick = np.sort(rng.choice(500, size=60, replace=False))
    got = recover_sample_idx(z + 1e-12 * rng.normal(size=500), att, z[pick], att[pick])
    assert got is not None and np.array_equal(np.sort(got), pick)


def test_recover_sample_idx_returns_none_when_not_found():
    z = np.arange(10, dtype=float)
    att = np.array(["A"] * 10)
    assert recover_sample_idx(z, att, np.array([3.5]), np.array(["A"])) is None  # không có dòng nào gần 3,5
    assert recover_sample_idx(z, att, np.array([3.0, 3.0]), np.array(["A", "A"])) is None  # hai dòng ghép vào một chỉ số


def test_explain_subsample_is_deterministic_sorted_stratified_subset():
    rng = np.random.default_rng(4)
    att = rng.choice(np.array(["DoS", "Exploits", "Fuzzers", "Backdoor"]), size=5000, p=[0.5, 0.3, 0.15, 0.05])
    explain_idx = rng.choice(5000, size=800, replace=False)          # thứ tự bất kỳ, như cột test_idx đã lưu
    a = explain_subsample(explain_idx, att, 120, seed=42)
    b = explain_subsample(np.sort(explain_idx)[::-1], att, 120, seed=42)
    assert np.array_equal(a, b) and np.array_equal(a, np.sort(a))     # không phụ thuộc thứ tự đầu vào, đã sắp xếp
    assert set(a) <= set(explain_idx) and len(a) == len(set(a))
    assert set(att[a]) == set(att[explain_idx])                       # mọi loại tấn công đều có mặt
    expected = stratified_pick(np.sort(explain_idx), att, 120, np.random.default_rng(43))
    assert np.array_equal(a, expected)                                # đúng quy tắc: stratified_pick với seed + 1
    assert not np.array_equal(a, explain_subsample(explain_idx, att, 120, seed=7))
