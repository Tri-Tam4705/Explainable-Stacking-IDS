import numpy as np
import pandas as pd

from xsids.data.cleaning import clean_dataset, downcast_numeric, feature_columns
from xsids.data.labels import add_family
from xsids.data.loader import _standardize
from xsids.data.synthetic import make_synthetic_netflow
from xsids.schema import ATTACK_COL, HARD_DROP_COLS, LABEL_COL, MINORITY_COL, WEIGHT_COL


def _prepared(cfg, n=8000, defects=True):
    raw = make_synthetic_netflow(n, "unsw", seed=3, defects=defects, include_ip=False)
    df, _ = _standardize(raw, "Label", "Attack")
    df, _ = add_family(df, cfg["families"])
    return df


def test_hard_columns_dropped(cfg):
    df, rep = clean_dataset(_prepared(cfg), cfg["cleaning"])
    assert not (set(HARD_DROP_COLS) & set(df.columns))
    assert set(rep["dropped_columns"]) == set(HARD_DROP_COLS) - {"IPV4_SRC_ADDR", "IPV4_DST_ADDR"}


def test_invalid_values_removed(cfg):
    df, rep = clean_dataset(_prepared(cfg), cfg["cleaning"])
    assert rep["nonfinite_rows"] > 0 and rep["out_of_range_rows"] > 0  # lỗi đã được phát hiện
    x = df[feature_columns(df)].to_numpy(dtype="float64")
    assert np.isfinite(x).all() and np.abs(x).max() <= cfg["cleaning"]["max_abs_value"]


def test_out_of_range_report_mode_keeps_rows_but_never_creates_inf(cfg):
    df = _prepared(cfg)
    out, rep = clean_dataset(df, dict(cfg["cleaning"], out_of_range="report"))
    col = out["DST_TO_SRC_AVG_THROUGHPUT"].to_numpy(dtype="float64")
    assert np.isfinite(col).all()  # không bị ép float32 thành inf
    assert col.max() > 1e300 and rep["invalid_rows_removed"] == rep["nonfinite_rows"]


def test_downcast_does_not_overflow_float32():
    d = pd.DataFrame({"a": [1.5, 1.0e300, 2.5], "b": [1.5, 2.5, 3.5], "c": [1.0, 2.0, 3.0]})
    out = downcast_numeric(d.copy(), ["a", "b", "c"])
    assert out["a"].dtype == np.float64 and np.isfinite(out["a"]).all()
    assert out["b"].dtype == np.float32
    assert pd.api.types.is_integer_dtype(out["c"])


def test_no_duplicate_vectors_after_clean(cfg):
    df, rep = clean_dataset(_prepared(cfg), cfg["cleaning"])
    assert rep["collapse"]["duplicate_rows_removed"] > 0
    assert not df.duplicated(subset=feature_columns(df)).any()


def test_weights_conserve_the_number_of_rows(cfg):
    df, rep = clean_dataset(_prepared(cfg), cfg["cleaning"])  # majority
    assert int(df[WEIGHT_COL].sum()) == rep["rows_after_invalid"]
    assert (df[WEIGHT_COL] >= 1).all()


def test_weighted_prevalence_stays_close_to_original(cfg):
    df, rep = clean_dataset(_prepared(cfg), cfg["cleaning"])
    assert abs(rep["attack_ratio_after_weighted"] - rep["attack_ratio_before"]) < 0.01
    assert rep["attack_ratio_after_unweighted"] > rep["attack_ratio_before"]  # gộp trùng làm lệch nếu không có trọng số


def test_majority_policy_keeps_label_and_attack_name_consistent(cfg):
    df, rep = clean_dataset(_prepared(cfg), cfg["cleaning"])
    assert rep["collapse"]["conflict_rows"] > 0 and df[MINORITY_COL].sum() > 0
    assert ((df[LABEL_COL] == 1) == (df[ATTACK_COL].astype(str) != "Benign")).all()


def test_drop_policy_removes_conflict_groups(cfg):
    df, rep = clean_dataset(_prepared(cfg), dict(cfg["cleaning"], conflict_policy="drop"))
    assert (df[MINORITY_COL] == 0).all()
    assert int(df[WEIGHT_COL].sum()) == rep["rows_after_invalid"] - rep["collapse"]["conflict_rows"]


def test_clean_data_untouched_when_no_defects(cfg):
    df, rep = clean_dataset(_prepared(cfg, defects=False), cfg["cleaning"])
    assert rep["nonfinite_rows"] == 0 and rep["out_of_range_rows"] == 0 and rep["collapse"]["conflict_rows"] == 0


def test_downcast_saves_memory(cfg):
    df = _prepared(cfg)
    before = df.memory_usage(deep=True).sum()
    out, _ = clean_dataset(df, cfg["cleaning"])
    assert out.memory_usage(deep=True).sum() < before


def _toy_vectors():
    rows = []
    # V1: 3 dòng "DoS" + 1 dòng "Fuzzers" (cùng nhãn tấn công): đại diện phải là "DoS", trọng số 4
    rows += [dict(f1=1, f2=1, label=1, attack="Fuzzers")] + [dict(f1=1, f2=1, label=1, attack="DoS")] * 3
    # V2: 2 dòng benign + 1 dòng "Exploits": đa số benign, 1 dòng tấn công bị xếp thiểu số
    rows += [dict(f1=2, f2=2, label=0, attack="Benign")] * 2 + [dict(f1=2, f2=2, label=1, attack="Exploits")]
    # V3: 1 dòng benign duy nhất
    rows += [dict(f1=3, f2=3, label=0, attack="Benign")]
    return pd.DataFrame(rows)


def test_collapse_picks_most_frequent_attack_name_not_first_row():
    from xsids.data.cleaning import collapse_duplicates

    out, info = collapse_duplicates(_toy_vectors(), ["f1", "f2"], "majority")
    v1 = out[out["f1"] == 1].iloc[0]
    assert v1[ATTACK_COL] == "DoS" and v1[WEIGHT_COL] == 4 and v1[LABEL_COL] == 1
    assert info["attack_name_ambiguous_groups"] == 1 and info["attack_name_ambiguous_rows"] == 4


def test_collapse_majority_benign_keeps_minority_accounting():
    from xsids.data.cleaning import collapse_duplicates

    out, info = collapse_duplicates(_toy_vectors(), ["f1", "f2"], "majority")
    v2 = out[out["f1"] == 2].iloc[0]
    assert v2[LABEL_COL] == 0 and v2[ATTACK_COL] == "Benign" and v2[WEIGHT_COL] == 3 and v2[MINORITY_COL] == 1
    assert info["conflict_rows"] == 3 and info["minority_rows_by_attack"] == {"Exploits": 1}
    assert int(out[WEIGHT_COL].sum()) == 8


def test_collapse_drop_policy_removes_conflict_vector():
    from xsids.data.cleaning import collapse_duplicates

    out, info = collapse_duplicates(_toy_vectors(), ["f1", "f2"], "drop")
    assert 2 not in set(out["f1"]) and len(out) == 2
