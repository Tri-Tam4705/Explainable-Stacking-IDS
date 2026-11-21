import numpy as np
import pandas as pd

from xsids.audit import audit_dataframe, render_markdown
from xsids.data.labels import add_family
from xsids.data.loader import read_netflow_csv
from xsids.data.pipeline import prepare_dataset
from xsids.data.synthetic import make_synthetic_netflow
from xsids.schema import HARD_DROP_COLS, LABEL_COL


def _write_csv(tmp_path, style="unsw", n=15000):
    path = tmp_path / f"{style}.csv"
    make_synthetic_netflow(n, style, seed=5).to_csv(path, index=False)
    return path


def test_loader_drops_ip_and_standardizes_labels(tmp_path):
    df, info = read_netflow_csv(_write_csv(tmp_path), "Label", "Attack")
    assert "IPV4_SRC_ADDR" not in df.columns
    assert {"label", "attack"} <= set(df.columns)
    assert df["label"].dtype == np.int8
    assert info["missing_feature_cols"] == []


def test_loader_sampling_keeps_prevalence(tmp_path):
    path = _write_csv(tmp_path, n=40000)
    full, _ = read_netflow_csv(path, "Label", "Attack", sample_frac=1.0)
    part, _ = read_netflow_csv(path, "Label", "Attack", sample_frac=0.3, chunksize=5000)
    assert 0.2 < len(part) / len(full) < 0.4
    assert abs(part["label"].mean() - full["label"].mean()) < 0.01


def test_audit_flags_shortcut_features(tmp_path, cfg):
    df, _ = read_netflow_csv(_write_csv(tmp_path), "Label", "Attack")
    df, fam = add_family(df, cfg["families"])
    rep = audit_dataframe(df, cfg["audit"], fam)
    assert "MIN_TTL" in rep["shortcut_suspects"]
    assert rep["duplicates_raw"] > 0 and rep["nonfinite"]["rows"] > 0
    assert "DST_TO_SRC_AVG_THROUGHPUT" in rep["implausible_large_values"]  # giá trị đệm bị phát hiện
    assert rep["value_ranges"]["IN_BYTES"]["max"] > 0
    assert isinstance(rep["saturation"], dict) and "Nghi lối tắt" in render_markdown(rep, "t")
    ic = rep["icmp_contamination"]
    assert ic["icmp_type_nonzero_on_non_icmp"] > 0 and ic["of_which_equal_to_retransmitted_in_bytes"] > 0
    pairs = {frozenset((a, b)) for a, b, _ in rep["high_corr_pairs"]}
    assert frozenset(("SRC_TO_DST_SECOND_BYTES", "SRC_TO_DST_AVG_THROUGHPUT")) in pairs
    assert "Báo cáo kiểm toán" in render_markdown(rep, "test")


def test_prepare_has_no_leakage_between_splits(tmp_path, cfg):
    manifest = prepare_dataset(_write_csv(tmp_path), tmp_path / "out", cfg, "synthetic")
    tr = pd.read_parquet(tmp_path / "out" / "train.parquet")
    va = pd.read_parquet(tmp_path / "out" / "val.parquet")
    te = pd.read_parquet(tmp_path / "out" / "test.parquet")

    feats = manifest["feature_columns"]
    assert not (set(HARD_DROP_COLS) & set(tr.columns))
    h = lambda d: set(pd.util.hash_pandas_object(d[feats], index=False).tolist())  # noqa: E731
    assert not (h(tr) & h(te)) and not (h(tr) & h(va)) and not (h(va) & h(te))

    n = len(tr) + len(va) + len(te)
    assert abs(len(te) / n - cfg["split"]["test_size"]) < 0.01
    ratios = [d[LABEL_COL].mean() for d in (tr, va, te)]
    assert max(ratios) - min(ratios) < 0.005
    assert (tmp_path / "out" / "manifest.json").exists()
    # tỷ lệ tấn công CÓ TRỌNG SỐ của tập test phải gần tỷ lệ tự nhiên ban đầu
    wr = manifest["summary"]["test"]["weighted_attack_ratio"]
    assert abs(wr - manifest["cleaning"]["attack_ratio_before"]) < 0.01


def test_prepare_works_for_cse_style(tmp_path, cfg):
    manifest = prepare_dataset(_write_csv(tmp_path, "cse"), tmp_path / "out_cse", cfg, "synthetic-cse")
    fam = manifest["family"]["mapping"]
    assert fam["DoS"] == "DoS" and fam["DDoS"] == "DoS" and fam["Benign"] == "Benign"
    assert manifest["family"]["label_family_inconsistent_rows"] >= 0


def test_peek_reports_schema_and_attack_names(tmp_path):
    from xsids.data.peek import peek_csv, render_peek

    path = _write_csv(tmp_path)
    rep = peek_csv(path, "Label", "Attack", nrows=5000, scan_attack=True)
    assert rep["expected_missing"] == [] and rep["has_label_col"] and rep["has_attack_col"]
    assert "Benign" in rep["attack_values_full_scan"] and "DoS" in rep["attack_values_full_scan"]
    assert "Cột nhãn" in render_peek(rep)


def test_dataset_path_override(cfg, tmp_path):
    from xsids.config import dataset_csv_path

    cfg2 = {**cfg, "datasets": {**cfg["datasets"], "unsw": {**cfg["datasets"]["unsw"], "path": str(tmp_path / "x.csv")}}}
    assert dataset_csv_path(cfg2, "unsw") == tmp_path / "x.csv"
    assert dataset_csv_path(cfg2, "unsw", raw_dir_override=str(tmp_path / "d")).parent == tmp_path / "d"
    assert dataset_csv_path(cfg, "cse").name == "NF-CSE-CIC-IDS2018-v2.csv"
