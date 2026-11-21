"""Bốn mô hình thành phần dạng cây và cách lấy đầu ra thô fₖ(x)."""
from __future__ import annotations

import numpy as np
import yaml
from pathlib import Path
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

from xsids.config import project_root

BASE_NAMES = ["rf", "et", "xgb", "lgbm"]
# đại lượng mà TreeSHAP giải thích cho từng mô hình, cũng là đầu vào của meta-learner
RAW_KIND = {"rf": "prob", "et": "prob", "xgb": "margin", "lgbm": "margin"}


def load_models_cfg(path=None) -> dict:
    path = Path(path) if path else project_root() / "configs" / "models.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_params(cfg: dict, fast: bool = False) -> dict:
    params = {k: dict(v) for k, v in cfg["params"].items()}
    if fast:
        for k, v in cfg["fast_params"].items():
            params[k].update(v)
    return params


def make_model(name: str, params: dict, seed: int, n_jobs: int):
    p = dict(params)
    if name == "rf":
        return RandomForestClassifier(random_state=seed, n_jobs=n_jobs, **p)
    if name == "et":
        return ExtraTreesClassifier(random_state=seed, n_jobs=n_jobs, **p)
    if name == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(random_state=seed, n_jobs=n_jobs, eval_metric="logloss", **p)
    if name == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(random_state=seed, n_jobs=n_jobs, **p)
    raise ValueError(name)


def raw_output(name: str, model, X) -> np.ndarray:
    """fₖ(x): xác suất (rf, et) hoặc margin/log-odds thô (xgb, lgbm)."""
    if name in ("rf", "et"):
        return model.predict_proba(X)[:, 1].astype("float64")
    if name == "xgb":
        return np.asarray(model.predict(X, output_margin=True), dtype="float64")
    if name == "lgbm":
        return np.asarray(model.predict(X, raw_score=True), dtype="float64")
    raise ValueError(name)


def to_proba(name: str, raw: np.ndarray) -> np.ndarray:
    return raw if RAW_KIND[name] == "prob" else 1.0 / (1.0 + np.exp(-raw))
