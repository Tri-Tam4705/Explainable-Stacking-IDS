"""Bước 28: các kiểm tra bổ sung theo góp ý của giảng viên.

1. Baseline ghép đều không cần học trọng số: trung bình đầu ra thô (giải thích chính xác được, lời giải thích chính là E2)
   và trung bình logit (RF, ET đổi xác suất sang logit). Ngưỡng chọn trên val như mọi mô hình khác.
2. Khoảng tin cậy bootstrap theo vector cho DR và FAR trên tập test (bootstrap Poisson, mỗi vector giữ trọng số n_orig),
   kèm chênh lệch ghép cặp stacking − từng mô hình.
3. Phân tích DDoS LOIC-HTTP trên CSE: số vector, độ tập trung trọng số, láng giềng gần nhất trong train, SHAP gộp mức nhóm.
4. Một ví dụ cảnh báo kèm lời giải thích mức nhóm và mức đặc trưng cho mỗi bộ dữ liệu.
"""
import argparse
import time
from pathlib import Path

import joblib
import numpy as np
from scipy.special import expit, logit

import _bootstrap  # noqa: F401
from xsids.config import load_config, resolve_path
from xsids.explain.shap_stack import explain_stack
from xsids.models.data import load_feature_set, load_split, natural_prevalence, xyw
from xsids.models.evaluate import best_threshold, evaluate
from xsids.models.stacking import raw_matrix, stack_logit
from xsids.models.zoo import BASE_NAMES, to_proba
from xsids.utils import get_logger, write_json

log = get_logger()
ap = argparse.ArgumentParser()
ap.add_argument("--dataset", choices=["unsw", "cse"], required=True)
ap.add_argument("--n-boot", type=int, default=1000)
args = ap.parse_args()
ds = args.dataset

cfg = load_config()
proc = resolve_path(cfg, "processed_dir")
rep = resolve_path(cfg, "reports_dir")
mdir = Path(cfg["_root"]) / "models" / ds
fs = load_feature_set(proc)
feats, groups = fs["features"], fs["groups"]
GROUPS = list(dict.fromkeys(groups[f] for f in feats))

tr, va, te = (load_split(proc, ds, s) for s in ("train", "val", "test"))
Xva, yva, wva, _ = xyw(va, feats)
Xte, yte, wte, ate = xyw(te, feats)
pi = natural_prevalence(proc, ds)
models = {n: joblib.load(mdir / f"{n}.joblib") for n in BASE_NAMES}
st = joblib.load(mdir / "stack.joblib")
w, b, thr_stack = np.asarray(st["w"]), float(st["b"]), float(st["threshold"])

t0 = time.time()
F_va, F_te = raw_matrix(models, Xva), raw_matrix(models, Xte)
log.info("Đầu ra thô val/test: %.0fs", time.time() - t0)


def to_logit(F):
    G = F.copy()
    for k, n in enumerate(BASE_NAMES):
        if n in ("rf", "et"):
            G[:, k] = logit(np.clip(F[:, k], 1e-6, 1 - 1e-6))
    return G


scores_va = {n: to_proba(n, F_va[:, k]) for k, n in enumerate(BASE_NAMES)}
scores_te = {n: to_proba(n, F_te[:, k]) for k, n in enumerate(BASE_NAMES)}
scores_va["stack_explainable"] = expit(stack_logit(F_va, w, b))
scores_te["stack_explainable"] = expit(stack_logit(F_te, w, b))
scores_va["mean_raw"], scores_te["mean_raw"] = F_va.mean(1), F_te.mean(1)
scores_va["mean_logit"], scores_te["mean_logit"] = to_logit(F_va).mean(1), to_logit(F_te).mean(1)

out = {"dataset": ds, "results": {}, "note": "mean_raw: trung bình đều đầu ra thô (xác suất RF, ET và margin XGB, LGBM). "
       "mean_logit: trung bình đều logit (RF, ET đổi xác suất sang logit, cắt ở 1e-6)."}
pred_te = {}
for k in scores_te:
    thr = thr_stack if k == "stack_explainable" else best_threshold(yva, scores_va[k], wva)
    r = evaluate(yte, scores_te[k], wte, thr, ate, pi)
    out["results"][k] = {m: r["weighted"][m] for m in ("dr", "far", "macro_f1", "pr_auc", "roc_auc", "precision_at_natural_prevalence")}
    out["results"][k]["threshold"] = thr
    out["results"][k]["per_attack_recall"] = {a: v["recall_weighted"] for a, v in r["per_attack_recall"].items()}
    pred_te[k] = (scores_te[k] >= thr).astype(np.int8)
    log.info("%-18s DR %.4f  FAR %.5f  PR-AUC %.4f", k, r["weighted"]["dr"], r["weighted"]["far"], r["weighted"]["pr_auc"])

# 2. bootstrap Poisson theo vector
rng = np.random.default_rng(2026)
att, ben = yte == 1, yte == 0
names = list(pred_te)
B, chunk = args.n_boot, 25
dr_b = {k: [] for k in names}
far_b = {k: [] for k in names}
wa, wb = wte[att], wte[ben]
Pa = np.stack([pred_te[k][att] for k in names]).astype("float64")
Pb = np.stack([pred_te[k][ben] for k in names]).astype("float64")
t0 = time.time()
for s in range(0, B, chunk):
    m = min(chunk, B - s)
    ca = rng.poisson(1.0, size=(m, att.sum())) * wa
    cb = rng.poisson(1.0, size=(m, ben.sum())) * wb
    dr = (ca @ Pa.T) / ca.sum(1, keepdims=True)
    fa = (cb @ Pb.T) / cb.sum(1, keepdims=True)
    for j, k in enumerate(names):
        dr_b[k].extend(dr[:, j].tolist())
        far_b[k].extend(fa[:, j].tolist())
log.info("Bootstrap %d lần: %.0fs", B, time.time() - t0)


def ci(v):
    v = np.asarray(v)
    return {"mean": float(v.mean()), "ci95": [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))], "sd": float(v.std(ddof=1))}


boot = {"n_boot": B, "method": "bootstrap Poisson theo vector, mỗi vector mang trọng số n_orig", "dr": {}, "far": {},
        "diff_stack_minus": {}}
for k in names:
    boot["dr"][k], boot["far"][k] = ci(dr_b[k]), ci(far_b[k])
    if k != "stack_explainable":
        d_dr = np.asarray(dr_b["stack_explainable"]) - np.asarray(dr_b[k])
        d_far = np.asarray(far_b["stack_explainable"]) - np.asarray(far_b[k])
        boot["diff_stack_minus"][k] = {"dr": ci(d_dr), "far": ci(d_far),
                                       "share_stack_dr_higher": float((d_dr > 0).mean())}
out["bootstrap"] = boot
# độ tập trung trọng số của tấn công: số vector chiếm 90% số luồng của từng loại
conc = {}
for a in np.unique(ate[att]):
    ww = np.sort(wte[(ate == a) & att])[::-1]
    c = np.cumsum(ww) / ww.sum()
    conc[str(a)] = {"n_vectors": int(len(ww)), "n_flows": int(ww.sum()), "top1_share": float(ww[0] / ww.sum()),
                    "n_vectors_for_90pct": int(np.searchsorted(c, 0.9) + 1)}
out["weight_concentration"] = conc


def group_phi(phi):
    return {g: phi[..., [i for i, f in enumerate(feats) if groups[f] == g]].sum(-1) for g in GROUPS}


# 3. DDoS LOIC-HTTP (CSE)
if ds == "cse":
    name = "DDoS attacks-LOIC-HTTP"
    m_all = (ate == name) & att
    miss = m_all & (pred_te["stack_explainable"] == 0)
    hit = m_all & (pred_te["stack_explainable"] == 1)
    info = {"n_vectors": int(m_all.sum()), "n_flows": int(wte[m_all].sum()),
            "missed_vectors": int(miss.sum()), "missed_flows": int(wte[miss].sum()),
            "missed_by_all_four_flows": int(wte[miss & np.all([pred_te[n] == 0 for n in BASE_NAMES], axis=0)].sum()),
            "missed_top_vector_flows": sorted(wte[miss].astype(int).tolist(), reverse=True)[:10],
            "z_missed_weighted_mean": float(np.average(stack_logit(F_te[miss], w, b), weights=wte[miss])) if miss.any() else None,
            "z_hit_weighted_mean": float(np.average(stack_logit(F_te[hit], w, b), weights=wte[hit])) if hit.any() else None,
            "logit_threshold": float(logit(thr_stack))}
    # so giá trị đặc trưng: trung vị có trọng số của luồng bỏ sót, luồng bắt được và luồng lành tính (test)
    def wmed(Xs, ws):
        o = np.argsort(Xs, axis=0)
        res = []
        for j in range(Xs.shape[1]):
            xs, ww = Xs[o[:, j], j], ws[o[:, j]]
            c = np.cumsum(ww)
            res.append(float(xs[np.searchsorted(c, c[-1] / 2)]))
        return res
    Xn = Xte.to_numpy("float64")
    info["median_missed"] = dict(zip(feats, wmed(Xn[miss], wte[miss])))
    info["median_hit"] = dict(zip(feats, wmed(Xn[hit], wte[hit])))
    bmask = ben.copy()
    sub = np.flatnonzero(bmask)
    sub = rng.choice(sub, size=min(200000, len(sub)), replace=False)
    info["median_benign"] = dict(zip(feats, wmed(Xn[sub], wte[sub])))
    # láng giềng gần nhất trong train (log1p rồi chuẩn hóa), 1 triệu vector lấy ngẫu nhiên
    from sklearn.neighbors import NearestNeighbors
    Xtr_df, ytr, wtr, atr = xyw(tr, feats)
    idx = rng.choice(len(Xtr_df), size=min(1_000_000, len(Xtr_df)), replace=False)
    A = np.log1p(np.clip(Xtr_df.to_numpy("float64")[idx], 0, None))
    mu, sd = A.mean(0), A.std(0) + 1e-9
    nn = NearestNeighbors(n_neighbors=10).fit((A - mu) / sd)
    def nn_benign_share(mask):
        Q = (np.log1p(np.clip(Xn[mask], 0, None)) - mu) / sd
        _, nb = nn.kneighbors(Q)
        lab, ww = ytr[idx][nb], wtr[idx][nb]
        share = ((lab == 0) * ww).sum(1) / ww.sum(1)
        return float(np.average(share, weights=wte[mask]))
    info["knn10_benign_weight_share_missed"] = nn_benign_share(miss)
    info["knn10_benign_weight_share_hit"] = nn_benign_share(hit)
    # SHAP gộp mức nhóm (trung bình có trọng số) trên tối đa 300 vector bỏ sót nặng nhất và 300 vector bắt được
    def shap_groups(mask, n=300):
        ii = np.flatnonzero(mask)
        ii = ii[np.argsort(-wte[ii])][:n]
        e = explain_stack(models, w, b, Xte.iloc[ii])
        gp = group_phi(e["phi_stack"])
        return {g: float(np.average(v, weights=wte[ii])) for g, v in gp.items()}, float(e["additivity_error"]["stack"].max())
    t0 = time.time()
    info["shap_group_missed"], e1 = shap_groups(miss)
    info["shap_group_hit"], e2 = shap_groups(hit)
    info["additivity_max"] = max(e1, e2)
    log.info("SHAP LOIC-HTTP: %.0fs", time.time() - t0)
    out["loic_http"] = info

# 4. ví dụ cảnh báo: vector tấn công bị bắt có trọng số lớn nhất của họ tấn công phổ biến nhất (theo số luồng)
fam = {"unsw": "Exploits", "cse": "DoS attacks-Hulk"}[ds]
cand = np.flatnonzero((ate == fam) & att & (pred_te["stack_explainable"] == 1))
i = int(cand[np.argmax(wte[cand])])
e = explain_stack(models, w, b, Xte.iloc[[i]])
phi = e["phi_stack"][0]
order = np.argsort(-np.abs(phi))
out["example_alert"] = {
    "attack": str(ate[i]), "n_orig": int(wte[i]), "z": float(e["z"][0]), "prob": float(expit(e["z"][0])),
    "threshold_prob": thr_stack, "threshold_logit": float(logit(thr_stack)), "base_value": e["base_stack"],
    "additivity_error": float(e["additivity_error"]["stack"][0]),
    "groups": {g: float(v[0]) for g, v in group_phi(e["phi_stack"]).items()},
    "top_features": [{"feature": feats[j], "group": groups[feats[j]], "value": float(Xte.iloc[i, j]), "phi": float(phi[j]),
                      "benign_median_train": float(np.median(tr.loc[tr["label"] == 0, feats[j]].to_numpy()))}
                     for j in order[:6]],
    "raw_per_base": {n: float(e["raw_per_base"][k][0]) for k, n in enumerate(BASE_NAMES)},
}
write_json(out, rep / f"review_checks_{ds}.json")
log.info("Đã ghi %s", rep / f"review_checks_{ds}.json")
