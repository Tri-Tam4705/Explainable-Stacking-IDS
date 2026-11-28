import json

import numpy as np

from xsids.explain.stability import bootstrap_rank_stability, kendall_tau_rows
from xsids.report.summaries import multiseed_text, render_all, shift_text, stability_text
from xsids.utils import write_json


def test_kendall_tau_rows_extremes_and_constant_rows():
    A = np.array([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], [5.0, 5.0, 5.0, 5.0]])
    B = np.array([[10.0, 20.0, 30.0, 40.0], [4.0, 3.0, 2.0, 1.0], [1.0, 2.0, 3.0, 4.0]])
    t = kendall_tau_rows(A, B)
    assert t[0] == 1.0 and t[1] == -1.0 and np.isnan(t[2])


def test_bootstrap_is_perfectly_stable_when_columns_are_well_separated():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(400, 5)) * 0.01 + np.array([5.0, 4.0, 3.0, 2.0, 1.0])   # thứ hạng cột rõ ràng
    r = bootstrap_rank_stability(A, 50, np.random.default_rng(1), top_k=2)
    assert abs(r["kendall_tau_mean"] - 1.0) < 1e-9 and r["kendall_tau_std"] < 1e-9
    assert r["topk_jaccard_mean"] == 1.0 and len(r["per_bootstrap"]) == 50 and r["n_columns"] == 5


def test_bootstrap_detects_unstable_ranking_and_ignores_sign():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(60, 8))                       # mọi cột cùng phân phối: thứ hạng toàn cục chỉ là nhiễu
    r = bootstrap_rank_stability(A, 100, np.random.default_rng(1))
    assert r["kendall_tau_mean"] < 0.9 and "topk_jaccard_mean" not in r
    same = bootstrap_rank_stability(-A, 100, np.random.default_rng(1))   # dùng |A| nên đảo dấu không đổi kết quả
    assert same["per_bootstrap"] == r["per_bootstrap"]


def test_bootstrap_is_reproducible_for_same_seed():
    A = np.random.default_rng(3).normal(size=(80, 6))
    a = bootstrap_rank_stability(A, 20, np.random.default_rng(7))
    b = bootstrap_rank_stability(A, 20, np.random.default_rng(7))
    assert a == b


def _stab(new=True):
    st = {"n_sample": 100, "n_bootstrap": 200, "fidelity": {"r2_group_to_z": 0.99},
          "stability": {"kendall_tau_mean": 0.95, "kendall_tau_std": 0.04},
          "e1_lgbm_single": {"agreement_with_stack_mean": 0.7, "agreement_with_stack_p50": 0.7},
          "e2_uniform_ensemble": {"agreement_with_stack_mean": 0.8, "agreement_with_stack_p50": 0.8}}
    if new:
        st["stability"]["level"] = "global_group_ranking"
        st["stability_feature"] = {"kendall_tau_mean": 0.91, "kendall_tau_std": 0.02, "n_columns": 25, "top_k": 5, "topk_jaccard_mean": 0.88}
        st["recompute_check"] = {"per_flow_kendall_tau_mean": 1.0, "matches_saved": True}
    return st


def test_stability_text_new_and_old_formats():
    t = stability_text("unsw", _stab())
    assert "thứ hạng nhóm toàn cục (200 bootstrap)" in t and "25 đặc trưng" in t and "Jaccard top-5" in t
    assert "kiểm tra tất định" in t and "CẢNH BÁO" not in t
    old = stability_text("unsw", _stab(new=False))
    assert "cách tính cũ" in old and "đặc trưng toàn cục" not in old
    bad = _stab()
    bad["recompute_check"]["matches_saved"] = False
    assert "CẢNH BÁO" in stability_text("unsw", bad)


def test_shift_and_multiseed_text_and_render_all_include_them():
    sh = {"features": [{"feature": "tcp_win_in", "ks_all": 0.6, "ks_benign": 0.7, "psi_all": 2.1, "psi_benign": 3.4},
                       {"feature": "in_pkts", "ks_all": 0.1, "ks_benign": 0.1, "psi_all": 0.05, "psi_benign": 0.04}],
          "cross": {"unsw->cse": {"prevalence": 0.12, "roc_auc": 0.15, "pr_auc": 0.06,
                                  "source_threshold": {"dr": 0.02, "far": 0.23}, "recalibrated_threshold": {"dr": 0.4, "far": 0.1}}}}
    t = shift_text(sh)
    assert "tcp_win_in" in t and "1/2 đặc trưng" in t and "UNSW → CSE" in t and "0.150" in t
    v = lambda m: {"mean": m, "std": 0.001, "values": [m, m]}  # noqa: E731
    ms = {"seeds": ["seed gốc", "seed1"], "metrics": {k: {m: v(0.9) for m in ("dr", "far", "macro_f1", "pr_auc")}
                                                      for k in ("rf", "stack_explainable")},
          "stack_vs_base": {"rf": {m: {"mean_diff": 0.0002, "std_diff": 0.0005, "n": 2, "a_better_in": 1}
                                   for m in ("dr", "far", "macro_f1", "pr_auc")}}}
    t2 = multiseed_text("unsw", ms)
    assert "NHIỀU SEED — UNSW" in t2 and "− Random Forest" in t2 and "1/2" in t2 and "+0.0002" in t2
    assert shift_text(None) == "" and multiseed_text("unsw", None) == ""
    empty = {"unsw": None}
    rep = {"datasets": ["unsw"], "baselines": dict(empty), "stacking": dict(empty), "explain": dict(empty),
           "stability": {"unsw": _stab()}, "ablation": dict(empty), "cross_lofo": None,
           "shift": sh, "multiseed": {"unsw": ms}}
    full = render_all(rep)
    assert "DỊCH CHUYỂN PHÂN PHỐI" in full and "NHIỀU SEED" in full and "ĐỘ ỔN ĐỊNH" in full


def test_write_json_replaces_nonfinite_with_null(tmp_path):
    p = tmp_path / "x.json"
    write_json({"a": float("nan"), "b": [np.float64("inf"), np.int64(3), np.bool_(True)], "c": np.array([1.0, np.nan])}, p)
    assert json.loads(p.read_text(encoding="utf-8")) == {"a": None, "b": [None, 3, True], "c": [1.0, None]}
