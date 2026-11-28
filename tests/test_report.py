import numpy as np
import pytest

from xsids.report.console import bar, best_marks, num, pct, table
from xsids.report.summaries import (
    collect_models,
    compare_text,
    cross_text,
    attack_type_text,
    group_shapley_text,
    label_audit_text,
    missed_text,
    models_text,
    multisplit_text,
    render_all,
    retrain_text,
)


def _model(dr, far):
    return {"weighted": {"dr": dr, "far": far, "macro_f1": 0.97, "pr_auc": 0.96, "precision_at_natural_prevalence": 0.93}}


def test_num_pct_bar_handle_missing_values():
    assert num(None) == "-" and num(float("nan")) == "-" and num(0.5, 2, sign=True) == "+0.50"
    assert pct(0.1234, 1) == "12.3%" and pct(None) == "-"
    assert bar(0.5, 1.0, 10) == "█████░░░░░" and bar(2.0, 1.0, 4) == "████" and bar(None, 1.0, 4) == "····"


def test_best_marks_ties_and_direction():
    assert best_marks([0.9, 0.95, 0.95, None]) == [False, True, True, False]
    assert best_marks([0.003, 0.001, 0.002], higher_is_better=False) == [False, True, False]
    assert best_marks([None, None]) == [False, False]


def test_table_columns_are_aligned_and_group_rows_span():
    t = table(["Tên", "Giá trị"], [["a", "1.0"], ["dài hơn nhiều", "22.50"], ["NHÓM"]])
    lines = t.splitlines()
    regular = [lines[0], lines[2], lines[3]]
    assert len({len(x) for x in regular}) == 1            # mọi hàng thường cùng độ rộng
    assert all(" │ " in x for x in regular) and "▸ NHÓM" in lines[4]


def test_models_text_marks_best_per_column():
    t = models_text("unsw", {"rf": _model(0.98, 0.003), "xgb": _model(0.97, 0.001)})
    rf_line = next(x for x in t.splitlines() if "Random Forest" in x)
    xgb_line = next(x for x in t.splitlines() if "XGBoost" in x)
    assert rf_line.count("★") >= 1 and "0.00100 ★" in xgb_line and "0.9800 ★" in rf_line


def test_collect_models_merges_sources_in_fixed_order():
    out = collect_models({"results": {"xgb": _model(0.9, 0.1), "rf": _model(0.9, 0.1)}},
                         {"results": {"stack_explainable": _model(0.9, 0.1)}})
    assert list(out) == ["rf", "xgb", "stack_explainable"]
    assert collect_models(None, None) == {}


def test_cross_text_reports_retained_dr():
    cl = {"cross_dataset": {"train_unsw_test_unsw": {"weighted": {"dr": 0.98, "far": 0.003, "macro_f1": 0.97, "pr_auc": 0.97}},
                            "train_unsw_test_cse": {"weighted": {"dr": 0.0238, "far": 0.23, "macro_f1": 0.41, "pr_auc": 0.06}},
                            "train_cse_test_cse": {"weighted": {"dr": 0.93, "far": 0.0001, "macro_f1": 0.98, "pr_auc": 0.95}}}}
    t = cross_text(cl)
    assert "UNSW → CSE" in t and "2.6%" in t               # 0,0238 / 0,93 giữ lại 2,6% DR của trong miền
    assert "trong miền ★" in t


def _rep():
    ds = ["unsw", "cse"]
    empty = {d: None for d in ds}
    return {"datasets": ds, "baselines": dict(empty), "stacking": dict(empty), "explain": dict(empty),
            "stability": dict(empty), "ablation": dict(empty), "cross_lofo": None}


def test_render_all_skips_missing_parts_without_crashing():
    t = render_all(_rep())
    assert "TỔNG HỢP KẾT QUẢ" in t and "SO SÁNH GIỮA CÁC BỘ DỮ LIỆU" not in t
    assert compare_text(_rep()) == ""


def test_compare_text_lists_model_explanation_and_faithfulness_rows():
    rep = _rep()
    rep["stacking"]["unsw"] = {"results": {"stack_explainable": _model(0.9842, 0.0027)}}
    rep["stability"]["unsw"] = {"stability": {"kendall_tau_mean": 0.95}, "stability_feature": {"kendall_tau_mean": 0.9}}
    rep["explain"]["unsw"] = {"agreement": {"mean": 0.81}}
    rep["group_shapley"] = {"unsw": {"full": {"deletion": {"1": {m: {"flip_rate": f} for m, f in
                                                                 (("shap_path", 0.1), ("group_shapley", 0.135),
                                                                  ("oracle", 0.156), ("random", 0.03))}}}}}
    t = compare_text(rep)
    assert "SO SÁNH GIỮA CÁC BỘ DỮ LIỆU" in t and "0.9842" in t and "0.950" in t
    assert "Shapley nhóm" in t and "15.6%" in t and "CSE" not in t


def test_extra_experiment_sections_render_values_and_skip_missing_or_fast_runs():
    m = {"dr": 0.98, "far": 0.003, "macro_f1": 0.977, "pr_auc": 0.979, "roc_auc": 0.99}
    ms = {"fast": False, "splits": [1, 2], "metrics": {"stack_explainable": {k: {"mean": v, "std": 0.001} for k, v in m.items()}},
          "stack_vs_base": {"rf": {k: {"mean_diff": 0.0066 if k == "pr_auc" else 0.0, "std_diff": 0.001, "n": 2,
                                       "p_value_paired_t": 0.0123, "a_better_in": 2} for k in m}}}
    t = multisplit_text("unsw", ms)
    assert "NHIỀU PHÉP CHIA — UNSW" in t and "+0.0066" in t and "0.0123" in t and "2/2" in t and "− Random Forest" in t
    s = {"stack_explainable": m}
    rt = {"configs": {"full": {"n_kept": 25, "in_domain": s, "cross": s},
                      "drop_x": {"n_kept": 24, "in_domain": {"stack_explainable": dict(m, macro_f1=0.975)}, "cross": s}}}
    t = retrain_text("cse", rt)
    assert "drop_x" in t and "-0.0020" in t and t.count("0.990") == 2
    dg = {"missed": {"dr": 0.93, "missed_mass": 10, "attack_mass": 100, "dr_without_top_attack": {"attack": "Infilteration", "dr": 0.98},
                     "by_attack": [{"attack": "Infilteration", "n_vectors": 5, "n_weighted": 10, "recall": 0.1, "missed_weighted": 9,
                                    "missed_share_of_attack_mass": 0.09, "missed_also_by_all_base_models": 1.0,
                                    "missed_in_label_conflict_groups": 0.0},
                                   {"attack": "Bot", "n_vectors": 2, "n_weighted": 90, "recall": 1.0, "missed_weighted": 0,
                                    "missed_share_of_attack_mass": 0.0, "missed_also_by_all_base_models": None,
                                    "missed_in_label_conflict_groups": None}]}}
    t = missed_text("cse", dg)
    assert "Infilteration" in t and "Bot" not in t.split("────")[-1].split("DR nếu")[0] and "0.9800" in t
    assert multisplit_text("unsw", dict(ms, fast=True)) == "" and retrain_text("cse", None) == "" and missed_text("cse", {}) == ""


def test_make_all_creates_figures_for_available_data_and_skips_the_rest(tmp_path):
    pytest.importorskip("matplotlib")
    from xsids.report.plots import make_all

    skipped = []
    rep = _rep()
    rep["explain_checks"] = {"unsw": _ec_json(), "cse": None}
    rep["sanity"] = {"unsw": None, "cse": None}
    made = make_all(rep, tmp_path, processed_dir=None, log=skipped.append)
    names = {p.name for p in made}
    assert {"fig_explain_checks.png", "fig_deletion_knn.png"} <= names and "fig_agreement.png" not in names
    assert all(p.exists() and p.stat().st_size > 5000 for p in made)
    assert "fig_ablation.png" not in names and any("fig_ablation" in m for m in skipped)   # thiếu dữ liệu nên bỏ qua, không lỗi
    assert not any("Error" in m for m in skipped)


def _repeat_json(n, key, ratio=None):
    rng = np.random.default_rng(n)
    base = {"dr": 0.98, "far": 0.003, "macro_f1": 0.977, "pr_auc": 0.975}
    metrics = {}
    for m in ["rf", "et", "xgb", "lgbm", "stack_explainable"]:
        metrics[m] = {}
        for q, v in base.items():
            vals = v + rng.normal(0, 0.001, n) + (0.006 if m == "stack_explainable" and q == "pr_auc" else 0)
            metrics[m][q] = {"values": list(vals), "mean": float(vals.mean()), "std": float(vals.std(ddof=1))}
    from xsids.models.experiment import stack_vs_base

    paired = stack_vs_base(metrics, list(base), test_train_ratio=ratio)
    return {"fast": False, key: list(range(1, n + 1)), "metrics": metrics, "stack_vs_base": paired}


def test_repeat_figures_and_text_render_from_multiseed_and_multisplit(tmp_path):
    pytest.importorskip("matplotlib")
    from xsids.report.plots import fig_repeat_runs

    rep = _rep()
    rep["multisplit"] = {"unsw": _repeat_json(10, "splits", 0.29), "cse": None}
    rep["multiseed"] = {"unsw": dict(_repeat_json(8, "seeds"), **{"seeds": [f"s{i}" for i in range(8)]}), "cse": None}
    p = fig_repeat_runs(rep, tmp_path)
    assert p is not None and p.exists() and p.stat().st_size > 20000
    t = multisplit_text("unsw", rep["multisplit"]["unsw"])
    assert "t hiệu chỉnh" in t and "Wilcoxon" in t and "10/10" in t        # PR-AUC: stacking hơn ở cả 10 phép chia
    rep["multisplit"]["unsw"]["fast"] = True
    assert fig_repeat_runs(rep, tmp_path) is None                          # lượt chạy thử không được vẽ


def _ec_json():
    from xsids.explain.checks import compare, pairwise

    rng = np.random.default_rng(0)
    A = rng.normal(size=(30, 8))
    B = A + rng.normal(0, 0.3, A.shape)
    c = compare(A, B, 5)
    lvl = {"shap_path": dict(c, group_top1_same=0.9), "shap_int": c, "median_abs_dz": 0.5, "still_flagged": 0.95}
    dele = {str(k): {m: {"mean_drop": 0.1 * k, "flip_rate": 0.1 * k} for m in ("shap_path", "shap_int", "random")} for k in (1, 3, 5)}
    knn = dict(dele, ood={"method": "shap_path", "k": 5, "nn_dist_original_median": 1.0, "nn_dist_median_fill_median": 3.0, "nn_dist_knn_fill_median": 1.1})
    return {"fast": False, "background_size": 500,
            "value": {"additivity_error_interventional": {"stack": 1e-6}, "feature": {"path_vs_int": c, "int_vs_int_other_bg": c},
                      "group": {"path_vs_int": compare(A[:, :6], B[:, :6], 1), "int_vs_int_other_bg": compare(A[:, :6], A[:, :6], 1)}},
            "knn": {"median_fill": dele, "knn_fill": knn,
                    "group": {g: {m: {"flip_rate": 0.1 * int(g), "mean_drop": 0.1} for m in ("shap_path", "shap_int", "random_features", "random_groups")} for g in ("1", "2")},
                    "full": {"n_flows": 30, "knn_fill": knn,
                             "group": {g: {"shap_path": {"flip_rate": 0.3 * int(g), "mean_drop": 0.2, "vs_random_features": {"mean_diff": 0.05, "ci95": [0.01, 0.09]}},
                                           "random_features": {"flip_rate": 0.05, "mean_drop": 0.1}, "random_groups": {"flip_rate": 0.02, "mean_drop": 0.1}}
                                       for g in ("1", "2")}}},
            "robust": {"levels": {"0.01": lvl, "0.05": lvl}},
            "seeds": {"models": [{}] * 3, "path": {"feature": pairwise([A, B, A], 5), "group": pairwise([A[:, :6], B[:, :6]], 1)}},
            "weighted": {"rank_weighted": ["packet_size", "volume"], "kendall": 1.0}}


def test_explain_check_figures_and_text_render(tmp_path):
    pytest.importorskip("matplotlib")
    from xsids.explain.checks import compare
    from xsids.report.plots import fig_deletion_knn, fig_explain_checks
    from xsids.report.summaries import explain_checks_text

    rng = np.random.default_rng(1)
    A = rng.normal(size=(30, 8))
    sn = {"fast": False, "roc_auc_random_labels": 0.51, "feature": compare(A, rng.normal(size=A.shape), 5),
          "group": compare(A[:, :6], rng.normal(size=(30, 6)), 1)}
    rep = _rep()
    rep["explain_checks"] = {"unsw": _ec_json(), "cse": None}
    rep["sanity"] = {"unsw": sn, "cse": None}
    for f in (fig_explain_checks, fig_deletion_knn):
        p = f(rep, tmp_path)
        assert p is not None and p.stat().st_size > 20000
    t = explain_checks_text("unsw", rep["explain_checks"]["unsw"], sn)
    assert "KIỂM TRA BỔ SUNG" in t and "Nhãn xáo trộn" in t and "láng giềng" in t
    assert "Xóa trọn nhóm" in t and "toàn mẫu giải thích, 30 luồng" in t and "LIME" not in t
    assert explain_checks_text("unsw", dict(_ec_json(), fast=True), sn) == ""


def test_group_shapley_text_renders_deletion_and_skips_fast_runs():
    meths = ("shap_path", "shap_int", "group_shapley", "oracle", "random")
    dele = {"n_flows": 271, **{g: {m: {"flip_rate": f} for m, f in zip(meths, (0.04, 0.05, 0.125, 0.162, 0.032))}
                               for g in ("1", "2")}}
    fc = {m: {"mean": v} for m, v in (("shap_path", 0.61), ("group_shapley", 0.71), ("ceiling_same_operator", 0.73))}
    ag = lambda s: {"spearman_signed_mean": s, "top1_signed_same": 0.9}  # noqa: E731
    gs = {"fast": False, "group": {"deletion": dele, "faithfulness_correlation": fc, "kernel_check_max_abs_diff": 4.2e-7,
                                   "stability_ref_B_vs_C": ag(0.93), "seconds_per_flow_one_table": 0.18}}
    t = group_shapley_text("unsw", gs)
    assert "SHAPLEY NHÓM TRONG PHÂN PHỐI — UNSW" in t and "Oracle (mức tối đa)" in t and "16.2%" in t
    assert "0.71" in t and "0.73" in t and "0.93" in t and "Kiểm tra chéo" in t and "LIME" not in t
    assert group_shapley_text("unsw", {**gs, "fast": True}) == "" and group_shapley_text("unsw", None) == ""


def test_data_limit_texts_render_and_skip_missing_or_fast():
    assert attack_type_text("cse", {"fast": True}) == "" and label_audit_text("cse", None) == ""
    pc = {"precision": 0.9, "recall": 0.8, "f1": 0.85, "n_vectors": 5, "n_weighted": 50}
    at = {"fast": False, "config": "lr0.05", "test": {"accuracy": 0.97, "macro_f1": 0.81,
          "per_class": {"Bot": pc, "Empty": {**pc, "n_vectors": 0}}}, "ceiling": {"accuracy_ceiling": 0.99},
          "end_to_end": {"overall": {"detected": 0.9, "detected_and_named": 0.88, "named_given_detected": 0.97}},
          "explain": {"Bot": {"top_group": "volume"}}}
    t = attack_type_text("cse", at)
    assert "lr0.05" in t and "Bot" in t and "volume" in t and "Empty" not in t and "88.0%" in t
    ew = {"weighted": {"dr": 0.93, "far": 0.00006, "pr_auc": 0.95}}
    la = {"fast": False, "overlap": {"overall": {"attack_recall_ceiling": 0.95},
                                     "by_attack": {"Infilteration": {"label": 1, "n_weighted": 100, "shared_with_other_label": 0.6,
                                                                     "unreachable": 0.5},
                                                   "Benign": {"label": 0, "n_weighted": 900, "shared_with_other_label": 0.01,
                                                              "unreachable": 0.0}}},
          "confident_learning": {"suspect_share_vectors": 0.02, "suspect_share_weighted": 0.01,
                                 "by_attack": {"Infilteration": {"suspect_share_weighted": 0.8}, "Benign": {"suspect_share_weighted": 0.001}}},
          "effect": {"n_test_suspects": 12, "main_model": {"test_all": ew, "test_without_suspects": ew},
                     "retrained_without_suspects": {"test_all": ew, "test_without_suspects": ew}}}
    t = label_audit_text("cse", la)
    assert "KIỂM TOÁN NHÃN — CSE" in t and "Infilteration" in t and "80.0%" in t and "95.00%" in t and "Huấn luyện lại" in t


def test_seed_pairs_reads_own_explain_checks():
    from xsids.report.plots import _seed_pairs

    assert _seed_pairs({"explain_checks": {"cse": {}}}, "cse") is None
    own = _seed_pairs({"explain_checks": {"unsw": _ec_json()}}, "unsw")
    assert own is not None and "spearman_flow_mean" in own
