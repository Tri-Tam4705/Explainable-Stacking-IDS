import numpy as np
import pandas as pd
import pytest

import xsids.models.determinism as det
from xsids.models.determinism import meta_weight_spread, oof_repeat_diff


def _data(n=600, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 5)), columns=list("abcde"))
    y = ((X["a"] + 0.5 * X["b"] + rng.normal(0, 0.5, n)) > 0.6).astype(int).to_numpy()
    att = np.where(y == 1, np.where(rng.random(n) < 0.5, "DoS", "Fuzzers"), "Benign")
    return X, y, att


def test_single_thread_random_forest_is_reproducible():
    X, y, att = _data()
    r = oof_repeat_diff("rf", {"n_estimators": 20, "max_depth": 5, "min_samples_leaf": 2, "max_features": "sqrt"},
                        X, y, att, folds=3, seed=1, n_jobs=1, repeats=2, ref_n_jobs=1)
    assert r["repeat"]["identical"] and r["repeat"]["n_rows_differ"] == 0
    assert r["reference"]["identical"] and len(r["runs"]) == 2 and len(r["runs"][0]) == len(X)


def test_detects_nondeterministic_model_and_reports_how_many_rows_differ(monkeypatch):
    X, y, att = _data()
    calls = {"n": 0}

    def noisy_oof(name, params, X, y, attack, folds, seed, n_jobs):
        calls["n"] += 1
        base = np.linspace(0, 1, len(X))
        out = base.copy()
        if calls["n"] > 1:                       # từ lần chạy thứ hai, một phần dự đoán lệch một chút
            out[:10] += 1e-5
        return out

    monkeypatch.setattr(det, "oof_raw", noisy_oof)
    r = oof_repeat_diff("xgb", {}, X, y, att, folds=3, seed=1, n_jobs=-1, repeats=2)
    assert not r["repeat"]["identical"]
    assert r["repeat"]["n_rows_differ"] == 10 and r["repeat"]["max_abs_diff"] == pytest.approx(1e-5)


def test_meta_weight_spread_zero_for_identical_inputs_and_positive_for_perturbed():
    X, y, _ = _data()
    F = np.column_stack([X["a"] + 0.1 * X["b"], X["a"] - 0.1 * X["c"], X["a"] + 0.05 * X["d"], X["a"]])
    same = meta_weight_spread([F, F.copy()], y, C=10.0)
    assert same["max_abs_spread"] == 0.0
    noisy = meta_weight_spread([F, F + np.random.default_rng(0).normal(0, 1e-2, F.shape)], y, C=10.0)
    assert noisy["max_abs_spread"] > 0 and noisy["weights"].shape == (2, 4)
