import numpy as np
import pandas as pd
import pytest

from xsids.explain.shap_stack import agreement_score, group_attribution
from xsids.models.evaluate import best_threshold, evaluate, precision_at_prevalence
from xsids.models.stacking import fit_meta, stack_logit
from xsids.models.zoo import BASE_NAMES


def test_precision_at_prevalence_matches_bayes_example():
    # TPR 0.99, FPR 0.01: pi = 1% -> 50%; pi = 3.98% -> ~80%
    assert abs(precision_at_prevalence(0.99, 0.01, 0.01) - 0.5) < 1e-9
    assert abs(precision_at_prevalence(0.99, 0.01, 0.0398) - 0.8036) < 1e-3


def test_evaluate_weighted_vs_unweighted_and_per_attack():
    y = np.array([1, 1, 0, 0, 0, 0])
    s = np.array([0.9, 0.2, 0.1, 0.1, 0.8, 0.05])
    w = np.array([1.0, 1.0, 100.0, 100.0, 1.0, 100.0])  # vector benign nặng bị xếp đúng
    att = np.array(["DoS", "Exploits", "Benign", "Benign", "Benign", "Benign"])
    r = evaluate(y, s, w, 0.5, att, 0.04)
    assert r["weighted"]["dr"] == 0.5 and r["unweighted"]["dr"] == 0.5
    assert r["weighted"]["far"] < r["unweighted"]["far"]  # trọng số làm FAR thấp vì benign nặng đúng
    assert r["per_attack_recall"]["DoS"]["recall_weighted"] == 1.0
    assert r["per_attack_recall"]["Exploits"]["recall_weighted"] == 0.0


def test_best_threshold_separates_perfectly_separable_scores():
    y = np.array([0] * 50 + [1] * 50)
    s = np.concatenate([np.linspace(0, 0.4, 50), np.linspace(0.6, 1, 50)])
    thr = best_threshold(y, s, np.ones(100))
    assert evaluate(y, s, np.ones(100), thr, np.array(["x"] * 100), 0.5)["weighted"]["f1_attack"] == 1.0


def test_best_threshold_is_robust_to_float_noise_on_tied_scores():
    # nhiều luồng cùng điểm 0.7 (như Extra Trees): ngưỡng phải nằm hẳn dưới 0.7 để lệch 1 ulp không đổi dự đoán
    y = np.array([0] * 40 + [1] * 60)
    s = np.concatenate([np.linspace(0.0, 0.3, 40), np.full(30, 0.7), np.linspace(0.8, 1.0, 30)])
    w = np.ones(100)
    thr = best_threshold(y, s, w)
    assert 0.3 < thr < 0.7
    for noisy in (np.nextafter(s, 0), np.nextafter(s, 1)):
        assert ((noisy >= thr) == (s >= thr)).all()


def test_best_threshold_gives_same_validation_predictions_as_the_best_score():
    rng = np.random.default_rng(3)
    y = (rng.random(400) < 0.3).astype(int)
    s = np.round(np.clip(0.35 * y + rng.normal(0.4, 0.2, 400), 0, 1), 2)  # điểm rời rạc, nhiều giá trị trùng
    w = rng.integers(1, 20, 400).astype(float)
    from sklearn.metrics import precision_recall_curve
    prec, rec, t = precision_recall_curve(y, s, sample_weight=w)
    f1 = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)
    assert ((s >= best_threshold(y, s, w)) == (s >= t[int(np.argmax(f1))])).all()


def test_meta_logit_is_linear_in_raw_outputs():
    rng = np.random.default_rng(1)
    F = rng.normal(size=(2000, 4))
    y = (F @ np.array([1.0, 0.5, -0.5, 2.0]) + rng.normal(size=2000) > 0).astype(int)
    w, b, lr = fit_meta(F, y, 10.0)
    assert np.allclose(stack_logit(F, w, b), lr.decision_function(F))


def test_group_attribution_conserves_total():
    phi = np.random.default_rng(2).normal(size=(10, 5))
    feats = ["a", "b", "c", "d", "e"]
    groups = {"a": "volume", "b": "volume", "c": "timing_rate", "d": "protocol", "e": "protocol"}
    G, names = group_attribution(phi, feats, groups, ["volume", "timing_rate", "protocol", "tcp_behavior"])
    assert names == ["volume", "timing_rate", "protocol"] and np.allclose(G.sum(axis=1), phi.sum(axis=1))


def test_agreement_score_extremes():
    g = np.random.default_rng(3).normal(size=(20, 6))
    assert np.allclose(agreement_score([g, g, g, g]), 1.0)
    assert np.allclose(agreement_score([g, -g]), -1.0)


def test_exact_shap_aggregation_satisfies_additivity():
    pytest.importorskip("shap")
    pytest.importorskip("xgboost")
    pytest.importorskip("lightgbm")
    from xsids.explain.shap_stack import explain_stack
    from xsids.models.stacking import raw_matrix
    from xsids.models.zoo import get_params, load_models_cfg, make_model

    rng = np.random.default_rng(4)
    X = pd.DataFrame(rng.normal(size=(1500, 6)), columns=list("abcdef"))
    y = ((X["a"] + 0.5 * X["b"] * X["c"] + rng.normal(0, 0.5, 1500)) > 0.3).astype(int).to_numpy()
    params = get_params(load_models_cfg(), fast=True)
    for k in params:
        params[k].update({"n_estimators": 20}) if "n_estimators" in params[k] else None
    models = {n: make_model(n, params[n], 0, 1).fit(X, y) for n in BASE_NAMES}
    w, b = np.array([0.7, 1.3, 0.4, 0.9]), -0.8
    ex = explain_stack(models, w, b, X.iloc[:60])
    for name, err in ex["additivity_error"].items():
        assert err.max() < 1e-3, (name, err.max())
    z_direct = stack_logit(raw_matrix(models, X.iloc[:60]), w, b)
    assert np.allclose(ex["z"], z_direct)
    assert np.allclose(ex["base_stack"] + ex["phi_stack"].sum(axis=1), z_direct, atol=1e-3)


def test_interventional_shap_aggregation_is_additive_with_a_shared_background():
    import numpy as np
    import pandas as pd

    from xsids.explain.shap_stack import explain_stack
    from xsids.models.experiment import fit_system

    rng = np.random.default_rng(3)
    n = 400
    y = (rng.random(n) < 0.3).astype(int)
    X = pd.DataFrame({"a": np.round(rng.normal(y * 2.0, 1.0), 1), "b": rng.normal(-y, 1.0), "c": rng.integers(0, 5, n).astype(float)})
    params = {"rf": {"n_estimators": 10, "max_depth": 4}, "et": {"n_estimators": 10, "max_depth": 4},
              "xgb": {"n_estimators": 10, "max_depth": 3}, "lgbm": {"n_estimators": 10, "num_leaves": 7, "verbose": -1}}
    s = fit_system(X, y, np.where(y == 1, "A", "B"), params, 0, 1, 3, 10.0)
    bg = X[y == 0].iloc[:50]
    ex = explain_stack(s["models"], s["w"], s["b"], X.iloc[:40], bg)
    assert max(v.max() for v in ex["additivity_error"].values()) < 1e-4   # kể cả khi giá trị trùng ngưỡng cắt (cột làm tròn)
