"""Stacking: dự đoán out-of-fold của 4 mô hình thành phần và meta-learner tuyến tính."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from xsids.data.split import make_strata
from xsids.models.zoo import BASE_NAMES, make_model, raw_output, to_proba


def oof_raw(name: str, params: dict, X, y, attack, folds: int, seed: int, n_jobs: int) -> np.ndarray:
    """fₖ out-of-fold: mỗi vector được dự đoán bởi mô hình chưa từng thấy nó."""
    strata = make_strata(pd.Series(attack), min_stratum_size=folds * 2)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    out = np.zeros(len(X), dtype="float64")
    for tr, te in skf.split(np.zeros(len(X)), strata):
        m = make_model(name, params, seed, n_jobs)
        m.fit(X.iloc[tr], y[tr])
        out[te] = raw_output(name, m, X.iloc[te])
    return out


def fit_meta(F: np.ndarray, y: np.ndarray, C: float):
    """Hồi quy logistic L2 trên đầu ra thô: z(x) = b + Σ wₖ·fₖ(x). Không chuẩn hóa để giữ nguyên tuyến tính."""
    lr = LogisticRegression(C=C, max_iter=5000)
    lr.fit(F, y)
    return lr.coef_.ravel().astype("float64"), float(lr.intercept_[0]), lr


def stack_logit(F: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return b + F @ w


def raw_matrix(models: dict, X) -> np.ndarray:
    return np.stack([raw_output(n, models[n], X) for n in BASE_NAMES], axis=1)


def proba_matrix(F: np.ndarray) -> np.ndarray:
    return np.stack([to_proba(n, F[:, k]) for k, n in enumerate(BASE_NAMES)], axis=1)
