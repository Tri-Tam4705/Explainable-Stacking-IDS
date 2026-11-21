import numpy as np
import pandas as pd

from xsids.data.cleaning import clean_dataset
from xsids.data.labels import add_family
from xsids.data.loader import _standardize
from xsids.data.synthetic import make_synthetic_netflow
from xsids.features.engineer import GROUP_ORDER, TCP_BITS, engineer
from xsids.features.select import build_feature_set, cluster_within_groups, screen_dataset
from xsids.schema import LABEL_COL, WEIGHT_COL

SCREEN = {"sparse_zero_threshold": 0.95, "shortcut_auc_threshold": 0.90, "corr_threshold": 0.90, "sample_rows": 20000}


def _clean(cfg, style="unsw", n=12000, seed=1):
    raw = make_synthetic_netflow(n, style, seed=seed, include_ip=False)
    df, _ = _standardize(raw, "Label", "Attack")
    df, _ = add_family(df, cfg["families"])
    df, _ = clean_dataset(df, cfg["cleaning"])
    return df


def test_tcp_flags_are_decoded_into_bits(cfg):
    df = _clean(cfg)
    X, groups = engineer(df)
    v = df["CLIENT_TCP_FLAGS"].to_numpy(dtype="int64")
    for bit, mask in TCP_BITS.items():
        assert (X[f"client_{bit}"].to_numpy() == ((v & mask) > 0)).all()
    assert groups["client_syn"] == "tcp_behavior"


def test_engineer_has_no_nonfinite_and_known_groups(cfg):
    X, groups = engineer(_clean(cfg))
    assert np.isfinite(X.to_numpy()).all()
    assert set(groups.values()) <= set(GROUP_ORDER) and set(groups) == set(X.columns)
    assert X.dtypes.eq("float32").all()


def test_screening_flags_constant_sparse_and_shortcut_columns():
    n = 4000
    rng = np.random.default_rng(0)
    y = (rng.random(n) < 0.3).astype(int)
    X = pd.DataFrame({"const": 1.0, "sparse": (rng.random(n) < 0.01).astype(float),
                      "leak": y + rng.normal(0, 0.01, n), "ok": rng.normal(size=n) + 0.3 * y})
    res = screen_dataset(X, y, np.ones(n), SCREEN)
    assert "hằng" in res["const"]["reasons"]
    assert any("thưa" in r for r in res["sparse"]["reasons"])
    assert any("lối tắt" in r for r in res["leak"]["reasons"])
    assert res["ok"]["reasons"] == []


def test_cluster_keeps_one_representative_per_correlated_pair():
    rng = np.random.default_rng(1)
    a = rng.normal(size=3000)
    pooled = pd.DataFrame({"a": a, "a_copy": a * 3 + 1, "b": rng.normal(size=3000)})
    groups = {"a": "volume", "a_copy": "volume", "b": "volume"}
    kept, clusters = cluster_within_groups(pooled, groups, {"a": 0.8, "a_copy": 0.7, "b": 0.6}, 0.9)
    assert sorted(kept) == ["a", "b"]  # a_copy bị gộp vào a (AUC cao hơn)


def test_build_feature_set_drops_shortcut_and_is_shared_across_datasets(cfg):
    per_ds = {}
    for d, style, seed in (("unsw", "unsw", 1), ("cse", "cse", 2)):
        df = _clean(cfg, style, seed=seed)
        X, groups = engineer(df)
        per_ds[d] = (X, df[LABEL_COL].to_numpy(), df[WEIGHT_COL].to_numpy(dtype="float64"))
    fs = build_feature_set(per_ds, groups, SCREEN)
    assert len(fs["features"]) >= 8 and set(fs["features"]).isdisjoint(fs["dropped"])
    assert all(f in X.columns for f in fs["features"])
    assert len(set(fs["groups"].values())) >= 3  # còn nhiều nhóm ngữ nghĩa
    for f in fs["features"]:
        assert all(not screens[f]["reasons"] for screens in fs["screening"].values())
