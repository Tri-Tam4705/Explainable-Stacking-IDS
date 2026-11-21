import numpy as np
import pytest

from xsids.report.console import bar, best_marks, num, pct, table
from xsids.report.summaries import (
    agreement_metrics,
    collect_models,
    compare_text,
    cross_text,
    deletion_ks,
    lime_text,
    missed_text,
    models_text,
    multisplit_text,
    relative_fidelity,
    render_all,
    retrain_text,
)


def _lime(shap5=0.79, lime5=0.74, rand5=0.32):
    dele = {"1": {m: {"mean_drop": v, "flip_rate": v} for m, v in (("shap", 0.6), ("lime", 0.5), ("random", 0.06))},
            "5": {m: {"mean_drop": v, "flip_rate": min(v, 1.0)} for m, v in (("shap", shap5), ("lime", lime5), ("random", rand5))},
            "n_eligible_mean": 18.4}
    summ = lambda m: {"mean": m, "p10": m - 0.1, "p50": m, "p90": m + 0.1, "n": 10}  # noqa: E731
    return {
        "dataset": "unsw", "target": "proba", "n_lime": 271, "num_perturbations": 5000, "discretizer": "quartile", "top_k": 5,
        "deletion_test": dele,
        "agreement_shap_lime": {"feature_spearman_abs": summ(0.38), "topk_jaccard": summ(0.35), "topk_sign_agreement": summ(0.71),
                                "group_spearman": summ(0.51), "top1_same_rate": 0.0},
        "global_importance": {"features": ["a", "b", "c"], "shap": [0.5, 0.3, 0.2], "lime": [0.6, 0.3, 0.1], "spearman": 0.74, "topk_overlap": 4},
        "stability_lime": {"n_flows": 50, "repeats": 5, "spearman": summ(0.75), "topk_jaccard": summ(0.78), "top1_same_rate": 1.0},
        "local_fidelity": {"lime_r2": summ(0.24), "frac_r2_below_0.5": 1.0, "local_pred_abs_error": summ(0.55)},
        "importance_by_family": {"DoS": {"n": 100, "shap_top": ["a", "b", "c"], "lime_top": ["a", "c", "b"], "overlap": 4, "k": 5}},
        "lime_top1_feature_counts": {"a": 271}, "shap_top1_feature_counts": {"b": 200, "a": 71},
    }


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


def test_relative_fidelity_matches_formula_and_guards_division():
    d = _lime()["deletion_test"]["5"]
    assert relative_fidelity(d) == pytest.approx((0.74 - 0.32) / (0.79 - 0.32))
    tie = {m: {"mean_drop": 0.3} for m in ("shap", "lime", "random")}
    assert relative_fidelity(tie) is None                  # SHAP không hơn ngẫu nhiên: không có nghĩa để chia
    assert relative_fidelity({}) is None
    assert deletion_ks(_lime()["deletion_test"]) == ["1", "5"]   # bỏ qua khóa n_eligible_mean


def test_agreement_metrics_shape_and_clipping():
    m = agreement_metrics(_lime(shap5=0.5, lime5=0.9, rand5=0.1))   # LIME vượt SHAP -> kẹp về 1,0
    assert len(m) == 10 and m[-1][1] == 1.0 and m[0][1] == 0.74


def test_lime_text_contains_key_sections_and_values():
    t = lime_text("unsw", _lime())
    for needle in ("TRUNG THỰC CỤC BỘ", "ĐỒNG THUẬN", "ỔN ĐỊNH", "MÔ HÌNH THAY THẾ", "DoS", "UNSW", "89%"):
        assert needle in t
    assert lime_text("unsw", None) == ""


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


def _rep(with_lime=True):
    ds = ["unsw", "cse"]
    empty = {d: None for d in ds}
    return {"datasets": ds, "baselines": dict(empty), "stacking": dict(empty), "explain": dict(empty),
            "stability": dict(empty), "ablation": dict(empty), "cross_lofo": None,
            "lime": {"unsw": _lime(), "cse": _lime(0.83, 0.38, 0.25)} if with_lime else dict(empty)}


def test_render_all_skips_missing_parts_without_crashing():
    t = render_all(_rep(with_lime=False))
    assert "TỔNG HỢP KẾT QUẢ" in t and "LIME so với SHAP" not in t
    full = render_all(_rep())
    assert "SO SÁNH GIỮA CÁC BỘ DỮ LIỆU" in full and full.count("LIME so với SHAP gộp") == 2
    assert compare_text(_rep(with_lime=False)) == ""


def test_extra_experiment_sections_render_values_and_skip_missing_or_fast_runs():
    m = {"dr": 0.98, "far": 0.003, "macro_f1": 0.977, "pr_auc": 0.979, "roc_auc": 0.99}
    ms = {"fast": False, "splits": [1, 2], "metrics": {"voting": {k: {"mean": v, "std": 0.001} for k, v in m.items()}},
          "stack_minus_voting": {k: {"mean_diff": 0.0066 if k == "pr_auc" else 0.0, "std_diff": 0.001, "n": 2,
                                     "p_value_paired_t": 0.0123, "a_better_in": 2} for k in m}}
    t = multisplit_text("unsw", ms)
    assert "NHIỀU PHÉP CHIA — UNSW" in t and "+0.0066" in t and "0.0123" in t and "2/2" in t
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
    made = make_all(_rep(), tmp_path, processed_dir=None, log=skipped.append)
    names = {p.name for p in made}
    assert "fig_agreement.png" in names
    assert all(p.exists() and p.stat().st_size > 5000 for p in made)
    assert "fig_ablation.png" not in names and any("fig_ablation" in m for m in skipped)   # thiếu dữ liệu nên bỏ qua, không lỗi
    assert not any("Error" in m for m in skipped)


def _repeat_json(n, key, ratio=None):
    rng = np.random.default_rng(n)
    base = {"dr": 0.98, "far": 0.003, "macro_f1": 0.977, "pr_auc": 0.975}
    metrics = {}
    for m in ["rf", "et", "xgb", "lgbm", "voting", "stack_explainable"]:
        metrics[m] = {}
        for q, v in base.items():
            vals = v + rng.normal(0, 0.001, n) + (0.006 if m == "stack_explainable" and q == "pr_auc" else 0)
            metrics[m][q] = {"values": list(vals), "mean": float(vals.mean()), "std": float(vals.std(ddof=1))}
    from xsids.models.experiment import paired_summary

    paired = {q: paired_summary(metrics["stack_explainable"][q]["values"], metrics["voting"][q]["values"],
                                lower_is_better=(q == "far"), test_train_ratio=ratio) for q in base}
    return {"fast": False, key: list(range(1, n + 1)), "metrics": metrics, "stack_minus_voting": paired}


def test_repeat_figures_and_text_render_from_multiseed_and_multisplit(tmp_path):
    pytest.importorskip("matplotlib")
    from xsids.report.plots import fig_repeat_diff, fig_repeat_runs

    rep = _rep()
    rep["multisplit"] = {"unsw": _repeat_json(10, "splits", 0.29), "cse": None}
    rep["multiseed"] = {"unsw": dict(_repeat_json(8, "seeds"), **{"seeds": [f"s{i}" for i in range(8)]}), "cse": None}
    for f in (fig_repeat_runs, fig_repeat_diff):
        p = f(rep, tmp_path)
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
    dele = {str(k): {m: {"mean_drop": 0.1 * k, "flip_rate": 0.1 * k} for m in ("shap_path", "shap_int", "lime", "random")} for k in (1, 3, 5)}
    knn = dict(dele, ood={"method": "shap_path", "k": 5, "nn_dist_original_median": 1.0, "nn_dist_median_fill_median": 3.0, "nn_dist_knn_fill_median": 1.1})
    return {"fast": False, "background_size": 500,
            "value": {"additivity_error_interventional": {"stack": 1e-6}, "feature": {"path_vs_int": c, "int_vs_int_other_bg": c, "lime_vs_path": c, "lime_vs_int": c},
                      "group": {"path_vs_int": compare(A[:, :6], B[:, :6], 1), "int_vs_int_other_bg": compare(A[:, :6], A[:, :6], 1)}},
            "knn": {"median_fill": dele, "knn_fill": knn,
                    "group": {g: {m: {"flip_rate": 0.1 * int(g), "mean_drop": 0.1} for m in ("shap_path", "lime", "random_features", "random_groups")} for g in ("1", "2")},
                    "full": {"n_flows": 30, "knn_fill": knn,
                             "group": {g: {"shap_path": {"flip_rate": 0.3 * int(g), "mean_drop": 0.2, "vs_random_features": {"mean_diff": 0.05, "ci95": [0.01, 0.09]}},
                                           "random_features": {"flip_rate": 0.05, "mean_drop": 0.1}, "random_groups": {"flip_rate": 0.02, "mean_drop": 0.1}}
                                       for g in ("1", "2")}}},
            "robust": {"levels": {"0.01": lvl, "0.05": lvl}, "lime": {"rel": 0.01, "input_and_seed": c, "seed_only": c}},
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
    assert "Xóa trọn nhóm" in t and "toàn mẫu giải thích, 30 luồng" in t
    assert explain_checks_text("unsw", dict(_ec_json(), fast=True), sn) == ""
