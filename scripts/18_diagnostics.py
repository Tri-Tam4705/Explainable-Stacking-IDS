"""Chẩn đoán hai hiện tượng cần giải thích trong báo cáo. Không ghi đè mô hình hay kết quả chính.

A. Trọng số meta-learner: vì sao một mô hình thành phần có trọng số ÂM (Random Forest trên UNSW)?
   Chạy lại OOF, rồi đo: tương quan và VIF giữa bốn đầu ra thô; trọng số khi học riêng từng mô hình; trọng số khi bỏ
   từng mô hình; meta-learner ràng buộc trọng số không âm; khoảng bootstrap của trọng số. Mỗi biến thể được đánh giá
   trên tập test (ngưỡng chọn lại trên val) để biết dấu âm có đóng góp gì cho hiệu năng không.

B. LIME: vì sao một đặc trưng luôn đứng hạng 1 (server_psh trên UNSW)?
   Đo trực tiếp trên mô hình, không qua LIME: với từng luồng đã được LIME giải thích, thay MỘT đặc trưng bằng giá trị rút
   từ tập train và ghi |Δz| trung bình. Đây là độ nhạy thật của mô hình với từng đặc trưng. Nếu đặc trưng hạng 1 của
   LIME cũng đứng đầu ở phép đo này thì LIME phản ánh đúng mô hình; nếu không thì thứ hạng đó là sản phẩm của LIME.

Dùng:
    python scripts/18_diagnostics.py --dataset unsw            # cả hai phần
    python scripts/18_diagnostics.py --dataset cse --part lime
Đầu ra: reports/diagnostics_<ds>.json và bảng in ra màn hình. Phần A mất khoảng thời gian của bước 4 (OOF).
"""
import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata, spearmanr

import _bootstrap  # noqa: F401
from xsids.config import load_config, resolve_path
from xsids.explain.lime_stack import binary_columns
from xsids.models.data import cap_train, load_feature_set, load_split, natural_prevalence, xyw
from xsids.models.evaluate import best_threshold, evaluate
from xsids.models.stacking import fit_meta, oof_raw, raw_matrix, stack_logit
from xsids.models.zoo import BASE_NAMES, get_params, load_models_cfg
from xsids.utils import get_logger, write_json

log = get_logger()
ap = argparse.ArgumentParser()
ap.add_argument("--dataset", choices=["unsw", "cse"], required=True)
ap.add_argument("--part", choices=["meta", "lime", "missed", "all"], default="all")
ap.add_argument("--fast", action="store_true")
ap.add_argument("--n-bootstrap", type=int, default=200, help="số lần bootstrap khi học lại meta-learner (phần A)")
ap.add_argument("--n-draws", type=int, default=100, help="số giá trị thay thế rút từ train cho mỗi đặc trưng (phần B)")
args = ap.parse_args()

cfg, mcfg = load_config(), load_models_cfg()
ds = args.dataset
proc, rdir = resolve_path(cfg, "processed_dir"), resolve_path(cfg, "reports_dir")
mdir = Path(cfg["_root"]) / "models" / ds
feats = load_feature_set(proc)["features"]
seed, n_jobs, folds, C = mcfg["seed"], mcfg["n_jobs"], mcfg["cv_folds"], mcfg["meta"]["C"]
models = {n: joblib.load(mdir / f"{n}.joblib") for n in BASE_NAMES}
st = joblib.load(mdir / "stack.joblib")
out_path = rdir / f"diagnostics_{ds}.json"
out = {"dataset": ds}
if out_path.exists():  # giữ phần kia nếu chỉ chạy lại một phần
    import json
    out.update(json.load(open(out_path, encoding="utf-8")))


def fit_nonneg(F, y, C):
    """Cùng hàm mục tiêu với LogisticRegression(C) nhưng ràng buộc w ≥ 0."""
    s = np.where(y == 1, 1.0, -1.0)

    def obj(t):
        w, b = t[:-1], t[-1]
        m = s * (F @ w + b)
        g = -s * expit(-m)
        return 0.5 * w @ w + C * np.logaddexp(0, -m).sum(), np.r_[w + C * (F.T @ g), C * g.sum()]

    r = minimize(obj, np.zeros(F.shape[1] + 1), jac=True, method="L-BFGS-B",
                 bounds=[(0, None)] * F.shape[1] + [(None, None)], options={"maxiter": 5000})
    return r.x[:-1], float(r.x[-1])


# ───────────────────────── A. trọng số meta-learner ─────────────────────────
if args.part in ("meta", "all"):
    t0 = time.time()
    params = get_params(mcfg, args.fast)
    tr = cap_train(load_split(proc, ds, "train"), mcfg["train_cap"].get(ds), seed)
    va, te = load_split(proc, ds, "val"), load_split(proc, ds, "test")
    Xtr, ytr, _, atr = xyw(tr, feats)
    Xva, yva, wva, _ = xyw(va, feats)
    Xte, yte, wte, ate = xyw(te, feats)
    pi = natural_prevalence(proc, ds)
    F = np.zeros((len(Xtr), len(BASE_NAMES)))
    for k, n in enumerate(BASE_NAMES):
        F[:, k] = oof_raw(n, params[n], Xtr, ytr, atr, folds, seed, n_jobs)
        log.info("OOF %s xong (%.0fs)", n, time.time() - t0)
    F_va, F_te = raw_matrix(models, Xva), raw_matrix(models, Xte)

    def test_metrics(w, b, cols):
        s_va, s_te = expit(stack_logit(F_va[:, cols], w, b)), expit(stack_logit(F_te[:, cols], w, b))
        r = evaluate(yte, s_te, wte, best_threshold(yva, s_va, wva), ate, pi)["weighted"]
        return {k: r[k] for k in ("dr", "far", "macro_f1", "pr_auc")}

    def variant(cols, nonneg=False):
        cols = list(cols)
        w, b = fit_nonneg(F[:, cols], ytr, C) if nonneg else fit_meta(F[:, cols], ytr, C)[:2]
        z = stack_logit(F[:, cols], w, b)
        ll = float(np.logaddexp(0, -np.where(ytr == 1, 1, -1) * z).mean())
        return {"weights": {BASE_NAMES[c]: float(v) for c, v in zip(cols, w)}, "bias": float(b),
                "oof_logloss": ll, "test": test_metrics(np.asarray(w), b, cols)}

    allc = list(range(len(BASE_NAMES)))
    variants = {"full": variant(allc), "nonneg": variant(allc, nonneg=True)}
    for k, n in enumerate(BASE_NAMES):
        variants[f"only_{n}"] = variant([k])
        variants[f"drop_{n}"] = variant([c for c in allc if c != k])

    pear = np.corrcoef(F, rowvar=False)
    spear = spearmanr(F).statistic
    vif = np.diag(np.linalg.inv(pear))
    sd = F.std(axis=0)
    w_full = np.array([variants["full"]["weights"][n] for n in BASE_NAMES])

    rng = np.random.default_rng(seed)
    boots = np.array([fit_meta(F[i], ytr[i], C)[0] for i in
                      (rng.integers(0, len(F), len(F)) for _ in range(args.n_bootstrap))])
    out["meta"] = {
        "n_train": int(len(F)), "names": BASE_NAMES, "saved_weights": {n: float(v) for n, v in zip(BASE_NAMES, st["w"])},
        "output_scale": {n: {"std": float(sd[k]), "min": float(F[:, k].min()), "max": float(F[:, k].max())}
                         for k, n in enumerate(BASE_NAMES)},
        "standardized_weights": {n: float(w_full[k] * sd[k]) for k, n in enumerate(BASE_NAMES)},
        "pearson": pear, "spearman": spear, "vif": {n: float(v) for n, v in zip(BASE_NAMES, vif)},
        "variants": variants,
        "bootstrap": {"n": args.n_bootstrap, "weights": {
            n: {"mean": float(boots[:, k].mean()), "p2.5": float(np.percentile(boots[:, k], 2.5)),
                "p97.5": float(np.percentile(boots[:, k], 97.5)), "frac_negative": float((boots[:, k] < 0).mean())}
            for k, n in enumerate(BASE_NAMES)}},
        "seconds": round(time.time() - t0, 1), "fast": args.fast,
    }

    m = out["meta"]
    print(f"\nA. TRỌNG SỐ META-LEARNER — {ds.upper()} ({len(F)} vector train, OOF {folds} fold)")
    print("─" * 96)
    print(f"  {'Mô hình':6s}{'w':>9s}{'w × độ lệch chuẩn':>20s}{'KTC 95% bootstrap':>24s}{'tỷ lệ âm':>10s}{'w khi học riêng':>17s}{'VIF':>8s}")
    for k, n in enumerate(BASE_NAMES):
        bw = m["bootstrap"]["weights"][n]
        ci = f"[{bw['p2.5']:+.3f}, {bw['p97.5']:+.3f}]"
        print(f"  {n:6s}{w_full[k]:>+9.3f}{m['standardized_weights'][n]:>+20.3f}{ci:>24s}{bw['frac_negative']:>10.0%}"
              f"{variants[f'only_{n}']['weights'][n]:>+17.3f}{vif[k]:>8.1f}")
    print("\n  Tương quan Pearson giữa bốn đầu ra thô (OOF):")
    print("        " + "".join(f"{n:>8s}" for n in BASE_NAMES))
    for i, n in enumerate(BASE_NAMES):
        print(f"  {n:6s}" + "".join(f"{pear[i, j]:>8.3f}" for j in range(len(BASE_NAMES))))
    print(f"\n  {'Biến thể':22s}{'log-loss OOF':>14s}{'DR':>9s}{'FAR':>10s}{'macro-F1':>10s}{'PR-AUC':>9s}   trọng số")
    for name, v in variants.items():
        if name.startswith("only_"):
            continue
        t = v["test"]
        ws = ", ".join(f"{a} {b:+.2f}" for a, b in v["weights"].items())
        print(f"  {name:22s}{v['oof_logloss']:>14.5f}{t['dr']:>9.4f}{t['far']:>10.5f}{t['macro_f1']:>10.4f}{t['pr_auc']:>9.4f}   {ws}")
    del tr, va, te

# ───────────────────── B. độ nhạy thật của mô hình với từng đặc trưng ─────────────────────
if args.part in ("lime", "all"):
    t0 = time.time()
    sp = proc / ds / "lime_sample.parquet"
    if not sp.exists():
        raise SystemExit(f"Chưa có {sp}; chạy bước lime trước.")
    L = pd.read_parquet(sp)
    te = load_split(proc, ds, "test")
    X = xyw(te, feats)[0].iloc[L["test_idx"].to_numpy()].to_numpy(dtype="float64")
    tr = load_split(proc, ds, "train")
    Xtr, ytr = xyw(tr, feats)[:2]
    bg = np.random.default_rng(seed + 2).choice(len(Xtr), size=min(mcfg["lime"]["background_size"], len(Xtr)), replace=False)
    Xbg, ybg = Xtr.iloc[bg].to_numpy(dtype="float64"), ytr[bg]
    w, b = st["w"], st["b"]

    def zf(A):
        return stack_logit(raw_matrix(models, pd.DataFrame(np.asarray(A, dtype="float32"), columns=feats)), w, b)

    n, d = X.shape
    z0 = zf(X)
    rng = np.random.default_rng(seed + 5)
    E = np.zeros((n, d))       # trung bình |Δz| khi thay đặc trưng j bằng giá trị rút từ train
    Echg = np.zeros((n, d))    # như trên nhưng chỉ tính các lần giá trị thật sự đổi
    for j in range(d):
        draws = Xbg[rng.integers(0, len(Xbg), args.n_draws), j]
        A = np.repeat(X, args.n_draws, axis=0)
        A[:, j] = np.tile(draws, n)
        dz = np.abs(zf(A) - np.repeat(z0, args.n_draws)).reshape(n, args.n_draws)
        chg = (np.tile(draws, n).reshape(n, args.n_draws) != X[:, [j]])
        E[:, j] = dz.mean(axis=1)
        Echg[:, j] = np.where(chg.any(axis=1), (dz * chg).sum(axis=1) / np.maximum(chg.sum(axis=1), 1), 0.0)
    lime = np.abs(L[[f"lime_{f}" for f in feats]].to_numpy())
    shap = np.abs(L[[f"shap_{f}" for f in feats]].to_numpy())

    def top1(M):
        j, c = np.unique(np.argmax(M, axis=1), return_counts=True)
        return {feats[a]: int(b) for a, b in sorted(zip(j, c), key=lambda t: -t[1])}

    def rho(A, B):
        ra, rb = rankdata(A, axis=1), rankdata(B, axis=1)
        ra, rb = ra - ra.mean(1, keepdims=True), rb - rb.mean(1, keepdims=True)
        return float(np.nanmean((ra * rb).sum(1) / np.sqrt((ra * ra).sum(1) * (rb * rb).sum(1))))

    lime_top = max(top1(lime).items(), key=lambda t: t[1])[0]
    jt = feats.index(lime_top)
    binary = [feats[i] for i in binary_columns(pd.DataFrame(Xbg, columns=feats))]
    rank_of = lambda M: (rankdata(-M, axis=1)[:, jt])  # noqa: E731  hạng của đặc trưng hạng 1 của LIME theo độ đo M
    vals, cnt = np.unique(X[:, jt], return_counts=True)
    out["lime_top1"] = {
        "n_flows": int(n), "n_draws": args.n_draws, "lime_top1_feature": lime_top, "is_binary": lime_top in binary,
        "binary_features": binary,
        "value_in_explained_flows": ({str(v): int(c) for v, c in zip(vals, cnt)} if len(vals) <= 10 else
                                     {"n_distinct": int(len(vals)), "median": float(np.median(X[:, jt]))}),
        "train_mean_benign": float(Xbg[ybg == 0, jt].mean()), "train_mean_attack": float(Xbg[ybg == 1, jt].mean()),
        "top1_counts": {"lime": top1(lime), "shap": top1(shap), "sensitivity": top1(E), "sensitivity_when_changed": top1(Echg)},
        "median_rank_of_lime_top1": {"lime": float(np.median(rank_of(lime))), "shap": float(np.median(rank_of(shap))),
                                     "sensitivity": float(np.median(rank_of(E))),
                                     "sensitivity_when_changed": float(np.median(rank_of(Echg)))},
        "mean_abs": {f: {"lime": float(lime[:, j].mean()), "shap": float(shap[:, j].mean()),
                         "sensitivity": float(E[:, j].mean()), "sensitivity_when_changed": float(Echg[:, j].mean())}
                     for j, f in enumerate(feats)},
        "per_flow_spearman": {"lime_vs_sensitivity": rho(lime, E), "shap_vs_sensitivity": rho(shap, E),
                              "lime_vs_sensitivity_when_changed": rho(lime, Echg),
                              "shap_vs_sensitivity_when_changed": rho(shap, Echg)},
        "seconds": round(time.time() - t0, 1),
    }
    q = out["lime_top1"]
    print(f"\nB. ĐỘ NHẠY THẬT CỦA MÔ HÌNH VỚI TỪNG ĐẶC TRƯNG — {ds.upper()} ({n} luồng, {args.n_draws} giá trị thay thế mỗi đặc trưng)")
    print("─" * 96)
    print(f"  Đặc trưng hạng 1 của LIME: {lime_top} ({'nhị phân' if q['is_binary'] else 'liên tục'}); "
          f"giá trị trong các luồng được giải thích: {q['value_in_explained_flows']}; "
          f"trung bình trên train: lành tính {q['train_mean_benign']:.3f}, tấn công {q['train_mean_attack']:.3f}")
    for lab, key in (("LIME", "lime"), ("SHAP gộp", "shap"), ("Độ nhạy (thay 1 đặc trưng)", "sensitivity"),
                     ("Độ nhạy, chỉ khi giá trị đổi", "sensitivity_when_changed")):
        tc = list(q["top1_counts"][key].items())[:4]
        print(f"  {lab:30s} hạng trung vị của {lime_top}: {q['median_rank_of_lime_top1'][key]:>4.0f}   hạng 1: "
              + ", ".join(f"{a} {c}/{n}" for a, c in tc))
    ps = q["per_flow_spearman"]
    print(f"  Spearman từng luồng với độ nhạy: LIME {ps['lime_vs_sensitivity']:.3f} · SHAP {ps['shap_vs_sensitivity']:.3f}"
          f"   (chỉ khi giá trị đổi: LIME {ps['lime_vs_sensitivity_when_changed']:.3f} · SHAP {ps['shap_vs_sensitivity_when_changed']:.3f})")
    order = np.argsort(-E.mean(axis=0))[:8]
    print(f"\n  {'Đặc trưng':28s}{'độ nhạy':>10s}{'khi đổi':>10s}{'|SHAP|':>10s}{'|LIME|':>10s}")
    for j in order:
        print(f"  {feats[j]:28s}{E[:, j].mean():>10.3f}{Echg[:, j].mean():>10.3f}{shap[:, j].mean():>10.3f}{lime[:, j].mean():>10.3f}")

# ───────────────────── C. tấn công bị bỏ sót thuộc loại nào ─────────────────────
if args.part in ("missed", "all"):
    import json as _json

    te = load_split(proc, ds, "test")
    X, y, wt, att = xyw(te, feats)
    F = raw_matrix(models, X)
    p = expit(stack_logit(F, st["w"], st["b"]))
    flagged = p >= st["threshold"]
    bpath = rdir / f"baselines_{ds}.json"
    base_thr = {n: v["threshold"] for n, v in _json.load(open(bpath, encoding="utf-8"))["results"].items()} if bpath.exists() else {}
    from xsids.models.zoo import to_proba
    by_any = np.zeros(len(te), dtype=bool)  # ít nhất một mô hình đơn gắn cờ (ở ngưỡng riêng của nó)
    for k, n in enumerate(BASE_NAMES):
        if n in base_thr:
            by_any |= to_proba(n, F[:, k]) >= base_thr[n]
    conflict = te["n_minority"].to_numpy() > 0 if "n_minority" in te.columns else np.zeros(len(te), dtype=bool)
    atk = y == 1
    tot = wt[atk].sum()
    rows = []
    for a in np.unique(att[atk]):
        m = atk & (att == a)
        miss = m & ~flagged
        wm = wt[miss].sum()
        rows.append({"attack": str(a), "n_vectors": int(m.sum()), "n_weighted": float(wt[m].sum()),
                     "recall": float(wt[m & flagged].sum() / wt[m].sum()), "missed_weighted": float(wm),
                     "missed_share_of_attack_mass": float(wm / tot),
                     "missed_also_by_all_base_models": float(wt[miss & ~by_any].sum() / wm) if wm else None,
                     "missed_in_label_conflict_groups": float(wt[miss & conflict].sum() / wm) if wm else None,
                     "median_p_of_missed": float(np.median(p[miss])) if miss.any() else None})
    rows.sort(key=lambda r: -r["missed_weighted"])
    miss_all = atk & ~flagged
    out["missed"] = {"threshold": float(st["threshold"]), "attack_mass": float(tot),
                     "missed_mass": float(wt[miss_all].sum()), "dr": float(1 - wt[miss_all].sum() / tot),
                     "dr_without_top_attack": None, "by_attack": rows}
    if rows:
        top = rows[0]["attack"]
        keep = atk & (att != top)
        out["missed"]["dr_without_top_attack"] = {"attack": top, "dr": float(wt[keep & flagged].sum() / wt[keep].sum())}
    print(f"\nC. TẤN CÔNG BỊ BỎ SÓT — {ds.upper()} (stacking, ngưỡng {st['threshold']:.4f}; DR {out['missed']['dr']:.4f})")
    print("─" * 110)
    print(f"  {'Loại tấn công':26s}{'vector':>8s}{'luồng':>9s}{'recall':>8s}{'bỏ sót':>9s}{'% tổng TC':>10s}"
          f"{'cả 4 mô hình bỏ sót':>21s}{'nhóm xung đột nhãn':>20s}")
    for r in rows:
        if r["missed_weighted"] < 0.5:
            continue
        print(f"  {r['attack']:26s}{r['n_vectors']:>8d}{r['n_weighted']:>9.0f}{r['recall']:>8.4f}{r['missed_weighted']:>9.0f}"
              f"{r['missed_share_of_attack_mass']:>10.1%}{r['missed_also_by_all_base_models']:>21.0%}"
              f"{r['missed_in_label_conflict_groups']:>20.0%}")
    d = out["missed"]["dr_without_top_attack"]
    if d:
        print(f"  DR nếu không tính loại '{d['attack']}': {d['dr']:.4f}")

write_json(out, out_path)
log.info("Đã ghi %s", out_path)
