"""Chỉ số đánh giá có trọng số (khôi phục tỷ lệ tấn công tự nhiên) và không trọng số."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def best_threshold(y, s, w) -> float:
    """Ngưỡng làm F1 (lớp tấn công, có trọng số) lớn nhất; chọn trên tập validation.

    Trả về điểm giữa của điểm số tốt nhất và điểm số liền dưới nó, không phải chính điểm số đó. Trên tập validation
    hai cách cho cùng dự đoán (không có điểm nào nằm giữa), nhưng điểm giữa không nhạy với sai số dấu phẩy động:
    nếu ngưỡng đúng bằng một điểm số, các luồng test có cùng điểm (thường gặp ở Random Forest / Extra Trees, vốn chỉ
    cho một số hữu hạn giá trị) sẽ đổi nhãn khi điểm lệch 1 đơn vị ở chữ số cuối giữa hai lần chạy.
    """
    prec, rec, thr = precision_recall_curve(y, s, sample_weight=w)
    f1 = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)
    i = int(np.argmax(f1))
    return float(thr[i]) if i == 0 else float((thr[i - 1] + thr[i]) / 2)


def _rates(y, pred, w):
    tp = w[(pred == 1) & (y == 1)].sum()
    fp = w[(pred == 1) & (y == 0)].sum()
    fn = w[(pred == 0) & (y == 1)].sum()
    tn = w[(pred == 0) & (y == 0)].sum()
    dr = tp / max(tp + fn, 1e-12)
    far = fp / max(fp + tn, 1e-12)
    prec = tp / max(tp + fp, 1e-12)
    f1a = 2 * prec * dr / max(prec + dr, 1e-12)
    prec_b = tn / max(tn + fn, 1e-12)
    rec_b = tn / max(tn + fp, 1e-12)
    f1b = 2 * prec_b * rec_b / max(prec_b + rec_b, 1e-12)
    return dict(dr=float(dr), far=float(far), precision=float(prec), f1_attack=float(f1a),
                macro_f1=float((f1a + f1b) / 2), tp=float(tp), fp=float(fp), fn=float(fn), tn=float(tn))


def precision_at_prevalence(dr: float, far: float, pi: float) -> float:
    """Xác suất một cảnh báo là tấn công thật khi tỷ lệ tấn công là pi (công thức Bayes)."""
    return float(dr * pi / max(dr * pi + far * (1 - pi), 1e-12))


def evaluate(y, s, w, thr: float, attack, pi: float) -> dict:
    pred = (s >= thr).astype(int)
    out = {"threshold": float(thr)}
    for tag, ww in (("weighted", w), ("unweighted", np.ones_like(w))):
        r = _rates(y, pred, ww)
        r["pr_auc"] = float(average_precision_score(y, s, sample_weight=ww))
        r["roc_auc"] = float(roc_auc_score(y, s, sample_weight=ww))
        r["precision_at_natural_prevalence"] = precision_at_prevalence(r["dr"], r["far"], pi)
        out[tag] = r
    per = {}
    for a in np.unique(attack[y == 1]):
        m = (attack == a) & (y == 1)
        per[str(a)] = {"n_vectors": int(m.sum()), "n_weighted": int(w[m].sum()),
                       "recall_weighted": float(w[m & (pred == 1)].sum() / max(w[m].sum(), 1e-12)),
                       "recall_unweighted": float((pred[m] == 1).mean())}
    out["per_attack_recall"] = per
    out["natural_prevalence_used"] = float(pi)
    return out
