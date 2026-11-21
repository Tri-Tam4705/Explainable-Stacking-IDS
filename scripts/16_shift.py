"""Phân tích vì sao đánh giá chéo bộ dữ liệu gần như thất bại (DR ≈ 0 khi huấn luyện ở bộ này, kiểm tra ở bộ kia).

Trả lời ba câu hỏi, đều không cần huấn luyện lại:
  1. Đặc trưng nào lệch phân phối nhiều nhất giữa UNSW và CSE? (KS và PSI; tính riêng cho luồng lành tính để tách
     "khác môi trường mạng" khỏi "khác loại tấn công")
  2. Khi xếp hạng (ROC-AUC, PR-AUC so với tỷ lệ tấn công) mô hình ở bộ này còn phân biệt được tấn công ở bộ kia không?
     PR-AUC thấp hơn tỷ lệ tấn công và ROC-AUC dưới 0,5 nghĩa là thứ tự điểm số bị đảo, không chỉ là ngưỡng lệch.
  3. Nếu chỉnh lại ngưỡng trên validation của bộ đích (tức có một ít nhãn đích), DR/FAR phục hồi được bao nhiêu?
     Nếu vẫn kém (DR thấp, hoặc DR ≈ 1 kèm FAR ≈ 1 vì ngưỡng tụt xuống mức gắn cờ mọi luồng) thì lỗi nằm ở phân phối
     đặc trưng chứ không phải ngưỡng.

Dùng:  python scripts/16_shift.py
Đầu ra: reports/shift.json và bảng in ra màn hình. Mất khoảng 1 phút.
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import ks_2samp

import _bootstrap  # noqa: F401
from xsids.config import load_config, resolve_path
from xsids.models.data import load_feature_set, load_split, natural_prevalence, xyw
from xsids.models.evaluate import best_threshold, evaluate
from xsids.models.stacking import raw_matrix, stack_logit
from xsids.models.zoo import BASE_NAMES
from xsids.utils import get_logger, write_json

log = get_logger()
ap = argparse.ArgumentParser()
ap.add_argument("--max-rows", type=int, default=200_000, help="số dòng val tối đa mỗi bộ để tính KS/PSI")
ap.add_argument("--bins", type=int, default=10)
ap.add_argument("--processed-dir", default=None)
args = ap.parse_args()

cfg = load_config()
proc = Path(args.processed_dir) if args.processed_dir else resolve_path(cfg, "processed_dir")
root, rdir = Path(cfg["_root"]), resolve_path(cfg, "reports_dir")
feats = load_feature_set(proc)["features"]
rng = np.random.default_rng(cfg.get("seed", 42))
DS = ["unsw", "cse"]


def psi(a: np.ndarray, b: np.ndarray, bins: int) -> float:
    """Population Stability Index, biên bin lấy theo phân vị của dữ liệu gộp (đối xứng giữa hai bộ)."""
    edges = np.unique(np.quantile(np.concatenate([a, b]), np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    ea, eb = np.histogram(a, edges)[0] / len(a), np.histogram(b, edges)[0] / len(b)
    ea, eb = np.clip(ea, 1e-4, None), np.clip(eb, 1e-4, None)
    return float(np.sum((ea - eb) * np.log(ea / eb)))


# ── 1. lệch phân phối từng đặc trưng ─────────────────────────────────────────
val = {}
for ds in DS:
    v = load_split(proc, ds, "val")
    X, y, _, _ = xyw(v, feats)
    if len(X) > args.max_rows:
        idx = np.sort(rng.choice(len(X), args.max_rows, replace=False))
        X, y = X.iloc[idx], y[idx]
    val[ds] = (X, y)

rows = []
for f in feats:
    r = {"feature": f}
    for tag, sel in (("all", None), ("benign", 0)):
        a = val["unsw"][0][f].to_numpy() if sel is None else val["unsw"][0][f].to_numpy()[val["unsw"][1] == 0]
        c = val["cse"][0][f].to_numpy() if sel is None else val["cse"][0][f].to_numpy()[val["cse"][1] == 0]
        r[f"ks_{tag}"] = float(ks_2samp(a, c).statistic)
        r[f"psi_{tag}"] = psi(a, c, args.bins)
    rows.append(r)
rows.sort(key=lambda r: r["psi_benign"], reverse=True)

# ── 2 và 3. xếp hạng và ngưỡng khi chéo bộ dữ liệu ──────────────────────────
cross = {}
for tr_ds in DS:
    models = {n: joblib.load(root / "models" / tr_ds / f"{n}.joblib") for n in BASE_NAMES}
    st = joblib.load(root / "models" / tr_ds / "stack.joblib")
    w, b = st["w"], st["b"]
    for te_ds in DS:
        Xv, yv, wv, _ = xyw(load_split(proc, te_ds, "val"), feats)
        Xt, yt, wt, at = xyw(load_split(proc, te_ds, "test"), feats)
        pi = natural_prevalence(proc, te_ds)
        pv = 1 / (1 + np.exp(-stack_logit(raw_matrix(models, Xv), w, b)))
        pt = 1 / (1 + np.exp(-stack_logit(raw_matrix(models, Xt), w, b)))
        src_thr = st["threshold"]
        r_src = evaluate(yt, pt, wt, src_thr, at, pi)["weighted"]
        r_rec = evaluate(yt, pt, wt, best_threshold(yv, pv, wv), at, pi)["weighted"]
        cross[f"{tr_ds}->{te_ds}"] = {
            "prevalence": pi, "roc_auc": r_src["roc_auc"], "pr_auc": r_src["pr_auc"],
            "source_threshold": {k: r_src[k] for k in ("dr", "far", "macro_f1")},
            "recalibrated_threshold": {k: r_rec[k] for k in ("dr", "far", "macro_f1")},
            "p_attack_mean_benign": float(np.average(pt[yt == 0], weights=wt[yt == 0])),
            "p_attack_mean_attack": float(np.average(pt[yt == 1], weights=wt[yt == 1])),
        }
write_json({"features": rows, "cross": cross, "psi_note": "PSI > 0,25 thường coi là lệch lớn; KS ∈ [0,1]"}, rdir / "shift.json")

# ── in kết quả ───────────────────────────────────────────────────────────────
print("\nLỆCH PHÂN PHỐI GIỮA UNSW VÀ CSE (tập val; xếp theo PSI của luồng lành tính)")
print("─" * 78)
print(f"  {'đặc trưng':28s} {'KS all':>8s} {'KS benign':>10s} {'PSI all':>9s} {'PSI benign':>11s}")
for r in rows[:12]:
    print(f"  {r['feature']:28s} {r['ks_all']:8.3f} {r['ks_benign']:10.3f} {r['psi_all']:9.2f} {r['psi_benign']:11.2f}")
n_big = sum(r["psi_benign"] > 0.25 for r in rows)
print(f"  → {n_big}/{len(rows)} đặc trưng có PSI(benign) > 0,25 (lệch lớn)")

print("\nĐÁNH GIÁ CHÉO: XẾP HẠNG VÀ NGƯỠNG (tập test đích, có trọng số)")
print("─" * 100)
print(f"  {'huấn luyện → kiểm tra':22s} {'tỷ lệ TC':>9s} {'ROC-AUC':>8s} {'PR-AUC':>7s} │ {'ngưỡng gốc DR/FAR':>20s} │ {'ngưỡng chỉnh lại DR/FAR':>24s}")
for k, v in cross.items():
    s, c = v["source_threshold"], v["recalibrated_threshold"]
    print(f"  {k:22s} {100 * v['prevalence']:8.2f}% {v['roc_auc']:8.3f} {v['pr_auc']:7.3f} │ {s['dr']:9.3f} / {s['far']:8.5f} │ {c['dr']:12.3f} / {c['far']:8.5f}")
print("  ROC-AUC ≈ 0,5 hoặc thấp hơn: mô hình không xếp hạng được tấn công ở bộ đích; PR-AUC < tỷ lệ TC: tệ hơn đoán ngẫu nhiên.")
print("  Ngưỡng chỉnh lại = ngưỡng F1 tốt nhất trên val của bộ đích. Nếu nó cho DR thấp, hoặc DR ≈ 1 kèm FAR ≈ 1 (gắn cờ mọi luồng),")
print("  thì không ngưỡng nào tách được tấn công: nguyên nhân là phân phối đặc trưng chứ không phải ngưỡng.")
log.info("Đã ghi reports/shift.json")
