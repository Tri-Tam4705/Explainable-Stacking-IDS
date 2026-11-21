import numpy as np
import pandas as pd

from xsids.data.split import make_strata, three_way_split
from xsids.schema import ATTACK_COL, LABEL_COL


def _toy(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    names = np.array(["Benign", "DoS", "Exploits", "Worms", "Rare1"])
    p = np.array([0.93, 0.04, 0.02, 0.0095, 0.0005])
    a = rng.choice(names, size=n, p=p)
    return pd.DataFrame({"x": rng.normal(size=n), ATTACK_COL: a, LABEL_COL: (a != "Benign").astype(int)})


def test_splits_are_disjoint_and_cover_everything():
    df = _toy()
    parts = three_way_split(df, 0.2, 0.1, seed=1)
    all_idx = np.concatenate(list(parts.values()))
    assert len(all_idx) == len(df) and len(set(all_idx)) == len(df)


def test_split_sizes_close_to_config():
    df = _toy()
    parts = three_way_split(df, 0.2, 0.1, seed=1)
    n = len(df)
    assert abs(len(parts["test"]) / n - 0.2) < 0.01
    assert abs(len(parts["val"]) / n - 0.1) < 0.01


def test_stratification_preserves_attack_ratio():
    df = _toy()
    parts = three_way_split(df, 0.2, 0.1, seed=1)
    overall = df[LABEL_COL].mean()
    for idx in parts.values():
        assert abs(df.iloc[idx][LABEL_COL].mean() - overall) < 0.005


def test_rare_classes_do_not_crash():
    df = _toy(n=3000)
    parts = three_way_split(df, 0.2, 0.1, seed=2, min_stratum_size=10)
    assert sum(len(v) for v in parts.values()) == len(df)


def test_make_strata_merges_small_classes():
    s = pd.Series(["A"] * 100 + ["B"] * 50 + ["C"] * 3 + ["D"] * 4)
    strata = make_strata(s, min_stratum_size=10)
    assert "C" not in set(strata) and "D" not in set(strata)


def test_split_is_reproducible():
    df = _toy()
    a = three_way_split(df, seed=7)
    b = three_way_split(df, seed=7)
    assert all((a[k] == b[k]).all() for k in a)


def _heavy_tailed(n=60000, seed=0):
    from xsids.schema import WEIGHT_COL

    rng = np.random.default_rng(seed)
    names = np.array(["Benign", "DoS", "Exploits", "Worms"])
    a = rng.choice(names, size=n, p=[0.9, 0.05, 0.04, 0.01])
    w = np.maximum(1, (rng.pareto(1.1, n) * 3)).astype("int64")  # đuôi rất nặng
    w[a == "Benign"] *= 5
    return pd.DataFrame({"x": rng.normal(size=n), ATTACK_COL: a, LABEL_COL: (a != "Benign").astype(int), WEIGHT_COL: w})


def test_weight_balanced_split_balances_weighted_prevalence_and_counts():
    from xsids.schema import WEIGHT_COL

    df = _heavy_tailed()
    w = df[WEIGHT_COL].to_numpy(float)
    glob = (w * df[LABEL_COL]).sum() / w.sum()
    parts = three_way_split(df, 0.2, 0.1, seed=3)
    assert sorted(np.concatenate(list(parts.values()))) == list(range(len(df)))
    for name, frac in (("train", 0.7), ("val", 0.1), ("test", 0.2)):
        sub = df.iloc[parts[name]]
        ws = sub[WEIGHT_COL].to_numpy(float)
        assert abs((ws * sub[LABEL_COL]).sum() / ws.sum() - glob) < 0.003  # lệch dưới 0,3 điểm %
        assert abs(len(sub) / len(df) - frac) < 0.02


def test_weight_balanced_split_beats_plain_random_split():
    from xsids.schema import WEIGHT_COL

    df = _heavy_tailed(seed=1)
    w = df[WEIGHT_COL].to_numpy(float)
    glob = (w * df[LABEL_COL]).sum() / w.sum()

    def spread(parts):
        r = [((df.iloc[i][WEIGHT_COL] * df.iloc[i][LABEL_COL]).sum() / df.iloc[i][WEIGHT_COL].sum()) for i in parts.values()]
        return max(abs(x - glob) for x in r)

    balanced = spread(three_way_split(df, 0.2, 0.1, seed=3))
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(df))
    tr, te = train_test_split(idx, test_size=0.2, random_state=3, stratify=df[ATTACK_COL])
    plain = spread({"train": tr, "test": te})
    assert balanced < plain


def test_weight_balanced_split_is_reproducible():
    df = _heavy_tailed()
    a, b = three_way_split(df, seed=9), three_way_split(df, seed=9)
    assert all((a[k] == b[k]).all() for k in a)
