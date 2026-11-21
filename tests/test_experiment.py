import numpy as np
import pandas as pd

from xsids.models.experiment import METRICS, evaluate_system, fit_system, paired_summary, scores, thresholds
from xsids.models.zoo import BASE_NAMES

PARAMS = {"rf": {"n_estimators": 10, "max_depth": 4}, "et": {"n_estimators": 10, "max_depth": 4},
          "xgb": {"n_estimators": 10, "max_depth": 3}, "lgbm": {"n_estimators": 10, "num_leaves": 7, "verbose": -1}}


def _data(n=600, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.3).astype(int)
    X = pd.DataFrame({"a": rng.normal(y * 2.0, 1.0), "b": rng.normal(-y * 1.5, 1.0), "noise": rng.normal(size=n)})
    att = np.where(y == 1, np.where(rng.random(n) < 0.5, "DoS", "Exploits"), "Benign")
    return X, y, np.ones(n), att


def test_paired_summary_counts_and_sign():
    r = paired_summary([0.9, 0.8, 0.7], [0.8, 0.7, 0.9])
    assert r["n"] == 3 and r["a_better_in"] == 2 and abs(r["mean_diff"] - 0.0) < 1e-12
    far = paired_summary([0.01, 0.02], [0.02, 0.04], lower_is_better=True)
    assert far["a_better_in"] == 2 and far["mean_diff"] < 0


def test_paired_summary_p_value_is_small_for_a_consistent_difference_and_nan_when_constant():
    a = np.array([0.90, 0.91, 0.92, 0.93, 0.94])
    assert paired_summary(a + 0.05 + np.array([0.001, -0.001, 0.002, 0.0, -0.002]), a)["p_value_paired_t"] < 0.001
    assert np.isnan(paired_summary(a + 0.05, a)["p_value_paired_t"])  # độ lệch chuẩn của chênh lệch bằng 0


def test_fit_and_evaluate_system_on_a_feature_subset():
    X, y, w, att = _data()
    Xv, yv, wv, _ = _data(300, 1)
    Xt, yt, wt, at = _data(300, 2)
    keep = ["a", "b"]  # bỏ một cột rồi huấn luyện lại
    system = fit_system(X[keep], y, att, PARAMS, seed=0, n_jobs=1, folds=3, C=10.0)
    assert set(system["models"]) == set(BASE_NAMES) and len(system["w"]) == len(BASE_NAMES)
    s = scores(system, Xt[keep])
    assert set(s) == set(BASE_NAMES) | {"voting", "stack_explainable"}
    assert all(((v >= 0) & (v <= 1)).all() for v in s.values())
    res = evaluate_system(system, thresholds(system, Xv[keep], yv, wv), Xt[keep], yt, wt, at, 0.3, per_attack=True)
    stack = res["stack_explainable"]
    assert all(m in stack for m in METRICS) and stack["roc_auc"] > 0.8
    assert set(stack["per_attack_recall"]) == {"DoS", "Exploits"}


def test_paired_summary_corrected_t_is_more_conservative_and_reports_ci_and_wilcoxon():
    rng = np.random.default_rng(0)
    b = rng.normal(0.95, 0.002, 10)
    a = b + 0.001 + rng.normal(0, 0.0008, 10)
    r = paired_summary(a, b, test_train_ratio=0.29)
    assert r["p_value_corrected_t"] > r["p_value_paired_t"]          # hiệu chỉnh làm p lớn hơn
    lo, hi = r["ci95"]
    clo, chi = r["ci95_corrected"]
    assert lo < r["mean_diff"] < hi and clo < lo and chi > hi          # khoảng hiệu chỉnh rộng hơn
    assert 0 < r["p_value_wilcoxon"] <= 1
    assert "p_value_corrected_t" not in paired_summary(a, b)            # chỉ tính khi có tỷ lệ test/train


def test_paired_summary_corrected_t_matches_the_nadeau_bengio_formula():
    from scipy.stats import t as tdist

    d = np.array([0.004, 0.006, 0.005, 0.007, 0.003])
    r = paired_summary(d, np.zeros(5), test_train_ratio=0.25)
    tc = d.mean() / (d.std(ddof=1) * np.sqrt(1 / 5 + 0.25))
    assert abs(r["p_value_corrected_t"] - 2 * tdist.sf(tc, 4)) < 1e-12
