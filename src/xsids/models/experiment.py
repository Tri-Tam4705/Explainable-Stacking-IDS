"""Huấn luyện và đánh giá trọn một hệ (4 mô hình đơn và stacking giải thích được) trên một tập đặc trưng bất kỳ.

Dùng cho các thí nghiệm phải huấn luyện lại nhiều lần (bỏ đặc trưng rồi huấn luyện lại, nhiều phép chia dữ liệu).
Cùng quy trình với scripts/06_baselines.py và 07_stacking.py: mô hình cuối huấn luyện trên toàn bộ train, meta-learner
học trên dự đoán out-of-fold, ngưỡng chọn trên val, chỉ số tính trên test có trọng số. Không lưu mô hình ra đĩa.
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit

from xsids.models.evaluate import best_threshold, evaluate
from xsids.models.stacking import fit_meta, oof_raw, raw_matrix, stack_logit
from xsids.models.zoo import BASE_NAMES, make_model, to_proba

METRICS = ("dr", "far", "macro_f1", "pr_auc", "roc_auc")


def fit_system(Xtr, ytr, atr, params: dict, seed: int, n_jobs: int, folds: int, C: float) -> dict:
    """Huấn luyện 4 mô hình trên toàn bộ train và meta-learner trên OOF. Trả về {models, w, b}."""
    models = {n: make_model(n, params[n], seed, n_jobs).fit(Xtr, ytr) for n in BASE_NAMES}
    F = np.column_stack([oof_raw(n, params[n], Xtr, ytr, atr, folds, seed, n_jobs) for n in BASE_NAMES])
    w, b, _ = fit_meta(F, ytr, C)
    return {"models": models, "w": w, "b": b}


def scores(system: dict, X) -> dict:
    """Điểm (xác suất tấn công) của 4 mô hình đơn và stacking trên X."""
    F = raw_matrix(system["models"], X)
    out = {n: to_proba(n, F[:, k]) for k, n in enumerate(BASE_NAMES)}
    out["stack_explainable"] = expit(stack_logit(F, system["w"], system["b"]))
    return out


def thresholds(system: dict, Xva, yva, wva) -> dict:
    return {k: best_threshold(yva, s, wva) for k, s in scores(system, Xva).items()}


def evaluate_system(system: dict, thr: dict, X, y, w, attack, pi: float, per_attack: bool = False) -> dict:
    """Chỉ số có trọng số của từng mô hình trên (X, y) với ngưỡng đã chọn."""
    out = {}
    for k, s in scores(system, X).items():
        r = evaluate(y, s, w, thr[k], attack, pi)
        out[k] = {m: r["weighted"][m] for m in METRICS}
        out[k]["threshold"] = r["threshold"]
        if per_attack:
            out[k]["per_attack_recall"] = {a: v["recall_weighted"] for a, v in r["per_attack_recall"].items()}
    return out


def paired_summary(a: np.ndarray, b: np.ndarray, lower_is_better: bool = False, test_train_ratio: float | None = None) -> dict:
    """Chênh lệch ghép cặp a − b qua các lần lặp.

    Trả về kiểm định t ghép cặp hai phía, kiểm định Wilcoxon dấu-hạng và khoảng tin cậy 95% của chênh lệch trung bình.
    Khi các lần lặp là các phép chia lại cùng một tập dữ liệu, tập train của các lần lặp chồng lên nhau nên các chênh lệch
    không độc lập và kiểm định t thông thường cho p quá nhỏ. Nếu truyền test_train_ratio = n_test / n_train, hàm tính thêm
    kiểm định t đã hiệu chỉnh của Nadeau và Bengio (2003): phương sai nhân với (1/n + n_test/n_train) thay cho 1/n.
    """
    from scipy.stats import t as tdist
    from scipy.stats import ttest_rel, wilcoxon

    d = np.asarray(a, dtype="float64") - np.asarray(b, dtype="float64")
    n = len(d)
    sd = float(d.std(ddof=1)) if n > 1 else float("nan")
    ok = n > 1 and sd > 0
    p = float(ttest_rel(a, b).pvalue) if ok else float("nan")
    out = {"mean_diff": float(d.mean()), "std_diff": sd, "n": int(n), "p_value_paired_t": p,
           "a_better_in": int((d < 0).sum() if lower_is_better else (d > 0).sum())}
    q = float(tdist.ppf(0.975, n - 1)) if n > 1 else float("nan")
    half = q * sd / np.sqrt(n) if ok else float("nan")
    out["ci95"] = [float(d.mean() - half), float(d.mean() + half)]
    try:
        out["p_value_wilcoxon"] = float(wilcoxon(d).pvalue) if ok and np.any(d != 0) else float("nan")
    except ValueError:
        out["p_value_wilcoxon"] = float("nan")
    if test_train_ratio is not None:
        se = sd * np.sqrt(1.0 / n + float(test_train_ratio)) if ok else float("nan")
        tc = d.mean() / se if ok else float("nan")
        out["test_train_ratio"] = float(test_train_ratio)
        out["p_value_corrected_t"] = float(2 * tdist.sf(abs(tc), n - 1)) if ok else float("nan")
        out["ci95_corrected"] = [float(d.mean() - q * se), float(d.mean() + q * se)] if ok else [float("nan")] * 2
    return out


def stack_vs_base(metrics: dict, metric_names=("dr", "far", "macro_f1", "pr_auc"), test_train_ratio: float | None = None) -> dict:
    """Chênh lệch ghép cặp Stacking − từng mô hình đơn qua các lần lặp: {mô hình: {chỉ số: paired_summary}}.

    metrics có dạng {mô hình: {chỉ số: {"values": [...]}}} như trong multiseed_*.json và multisplit_*.json.
    """
    if "stack_explainable" not in metrics:
        return {}
    out = {}
    for n in BASE_NAMES:
        if n not in metrics:
            continue
        out[n] = {q: paired_summary(metrics["stack_explainable"][q]["values"], metrics[n][q]["values"],
                                    lower_is_better=(q == "far"), test_train_ratio=test_train_ratio)
                  for q in metric_names if q in metrics[n] and q in metrics["stack_explainable"]}
    return out
