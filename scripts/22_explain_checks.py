"""Kiểm tra bổ sung cho lời giải thích, dùng mô hình chính (models/<ds>/) và các mô hình nhiều seed (models/<ds>_seed<k>/).

Các phần (chọn bằng --parts, mặc định chạy hết):
  value    : SHAP gộp với hai hàm giá trị. path-dependent (mặc định của đồ án, mỗi mô hình một "nền" riêng) so với
             interventional với CÙNG một tập nền lành tính cho cả bốn mô hình (khi đó SHAP gộp là đúng SHAP
             interventional của cả hệ). Đo thêm độ nhạy theo cách rút tập nền (hai lần rút khác nhau).
  knn      : phép xóa đặc trưng trong phân phối (thay bằng giá trị của luồng lành tính gần nhất) cho SHAP path,
             SHAP interventional, LIME và ngẫu nhiên; kèm phép xóa bằng trung vị lành tính để đối chiếu. Thêm phép xóa
             theo NHÓM đặc trưng (xóa trọn 1 hoặc 2 nhóm đóng góp lớn nhất), và chạy lại SHAP path so với ngẫu nhiên trên
             toàn bộ mẫu giải thích của bước 08. Mỗi so sánh có chênh lệch ghép cặp theo luồng và khoảng tin cậy 95%.
  robust   : độ bền cục bộ. Nhân mỗi đặc trưng liên tục với (1 + u), u ~ U(−δ, δ), rồi so lời giải thích trước và sau.
             LIME được đo cả độ dao động do nhiễu đầu vào và độ dao động chỉ do seed, để tách hai nguồn.
  seeds    : độ ổn định khi huấn luyện lại: SHAP gộp trên cùng một tập luồng với 8 mô hình (seed 42 và 1 đến 7).
  weighted : thứ hạng toàn cục có trọng số khôi phục tỷ lệ của dòng cảnh báo thật, so với trung bình không trọng số.

Dùng:
    python scripts/22_explain_checks.py --dataset unsw
    python scripts/22_explain_checks.py --dataset cse --parts value knn --fast
Cần: bước 08 (explain_sample.parquet), 12 (lime_sample.parquet); phần seeds cần models/<ds>_seed<k>/ của bước 17.
Đầu ra: reports/explain_checks_<ds>.json, data/processed/<ds>/explain_int_sample.parquet. Không ghi đè kết quả chính.
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from xsids.config import load_config, resolve_path
from xsids.explain.checks import (
    NeighborImputer,
    compare,
    knn_group_deletion_test,
    kendall,
    global_importance,
    knn_deletion_test,
    pairwise,
    perturb,
    pick_common,
    stratum_weights,
)
from xsids.explain.lime_stack import (
    binary_columns,
    build_explainer,
    deletion_test,
    lime_weights,
    make_lime_predict_fn,
    make_proba_fn,
    stratified_pick,
)
from xsids.explain.shap_stack import INTERVENTIONAL_BG_MAX, explain_stack, group_attribution
from xsids.features.engineer import GROUP_ORDER
from xsids.models.data import load_feature_set, load_split, xyw
from xsids.models.stacking import raw_matrix, stack_logit
from xsids.models.zoo import BASE_NAMES, load_models_cfg
from xsids.schema import ATTACK_COL, WEIGHT_COL
from xsids.utils import get_logger, write_json

PARTS = ["value", "knn", "robust", "seeds", "weighted"]
log = get_logger()
ap = argparse.ArgumentParser()
ap.add_argument("--dataset", choices=["unsw", "cse"], required=True)
ap.add_argument("--parts", nargs="+", choices=PARTS, default=PARTS)
ap.add_argument("--bg", type=int, default=500, help="số luồng lành tính rút từ train; SHAP interventional lấy mẫu tất định tối đa 100 luồng trong số này làm tập nền")
ap.add_argument("--knn-ref", type=int, default=20000, help="số luồng lành tính của train để tìm láng giềng")
ap.add_argument("--knn-ks", type=int, nargs="+", default=[1, 3, 5, 8, 10], help="số đặc trưng bị xóa trong phép xóa")
ap.add_argument("--knn-groups", type=int, nargs="+", default=[1, 2], help="số nhóm đặc trưng bị xóa")
ap.add_argument("--n-seed-flows", type=int, default=300)
ap.add_argument("--n-robust", type=int, default=100)
ap.add_argument("--robust-reps", type=int, default=5)
ap.add_argument("--robust-rel", type=float, nargs="+", default=[0.01, 0.05])
ap.add_argument("--fast", action="store_true", help="tham số nhỏ để chạy thử")
args = ap.parse_args()
if args.fast:
    args.bg, args.knn_ref, args.n_seed_flows, args.n_robust, args.robust_reps = 50, 2000, 20, 8, 2
    args.knn_ks, args.knn_groups = [1, 3], [1]

cfg, mcfg = load_config(), load_models_cfg()
ds, seed, K = args.dataset, mcfg["seed"], mcfg["lime"]["top_k"]
root, proc, rdir = Path(cfg["_root"]), resolve_path(cfg, "processed_dir"), resolve_path(cfg, "reports_dir")
fs = load_feature_set(proc)
feats, groups = fs["features"], fs["groups"]
lc = dict(mcfg["lime"])
lc.update((mcfg.get("lime_per_dataset") or {}).get(ds) or {})
ks = lc["deletion_ks"]


def load_system(mdir: Path):
    st = joblib.load(mdir / "stack.joblib")
    return {n: joblib.load(mdir / f"{n}.joblib") for n in BASE_NAMES}, np.asarray(st["w"]), float(st["b"]), float(st["threshold"])


models, w, b, thr = load_system(root / "models" / ds)
te = load_split(proc, ds, "test")
X, y, wt, att = xyw(te, feats)
z_all = stack_logit(raw_matrix(models, X), w, b)
flagged = np.flatnonzero(1 / (1 + np.exp(-z_all)) >= thr)
es = pd.read_parquet(proc / ds / "explain_sample.parquet")
ls = pd.read_parquet(proc / ds / "lime_sample.parquet")
lime_idx = ls["test_idx"].to_numpy(dtype=int)
assert np.allclose(ls["z"].to_numpy(), z_all[lime_idx], atol=1e-6), "lime_sample.parquet không khớp mô hình hiện tại: chạy lại bước 12"
phi_path = ls[[f"shap_{f}" for f in feats]].to_numpy(dtype="float64")
W_lime = ls[[f"lime_{f}" for f in feats]].to_numpy(dtype="float64")
Xl = X.iloc[lime_idx]

tr = load_split(proc, ds, "train")
Xtr, ytr, _, _ = xyw(tr, feats)
ben_idx = np.flatnonzero(ytr == 0)
bg1 = Xtr.iloc[np.random.default_rng(seed + 10).choice(ben_idx, min(args.bg, len(ben_idx)), replace=False)]
bg2 = Xtr.iloc[np.random.default_rng(seed + 11).choice(ben_idx, min(args.bg, len(ben_idx)), replace=False)]
# nền và giá trị "lành tính điển hình" của LIME: giống hệt bước 12
lime_bg_idx = np.random.default_rng(seed + 2).choice(len(Xtr), size=min(lc["background_size"], len(Xtr)), replace=False)
Xbg_lime, ybg_lime = Xtr.iloc[lime_bg_idx].reset_index(drop=True), ytr[lime_bg_idx]
baseline = Xbg_lime[ybg_lime == 0].median().to_numpy(dtype="float64")
bin_cols = binary_columns(Xtr)

out_path = rdir / f"explain_checks_{ds}.json"
out = json.load(open(out_path, encoding="utf-8")) if out_path.exists() else {}
if out.get("fast", args.fast) != args.fast:
    out = {}
out.update({"dataset": ds, "fast": args.fast, "background_drawn": int(len(bg1)), "background_size": int(min(len(bg1), INTERVENTIONAL_BG_MAX)), "n_lime_flows": int(len(lime_idx)),
            "features": feats, "groups": GROUP_ORDER})


def groups_of(phi):
    return group_attribution(phi, feats, groups, GROUP_ORDER)[0]


def save():
    write_json(out, out_path)


t_all = time.time()
phi_int = None
int_path = proc / ds / "explain_int_sample.parquet"

# ── value: path-dependent so với interventional ─────────────────────────────
if "value" in args.parts or "knn" in args.parts or "robust" in args.parts:
    t0 = time.time()
    ex1 = explain_stack(models, w, b, Xl, bg1)
    phi_int = ex1["phi_stack"]
    err = {k: float(v.max()) for k, v in ex1["additivity_error"].items()}
    log.info("SHAP interventional (nền %d luồng lành tính): %.0fs, sai số tính cộng lớn nhất %.1e", min(len(bg1), INTERVENTIONAL_BG_MAX), time.time() - t0, max(err.values()))
    pd.DataFrame({"test_idx": lime_idx, **{f"int_{f}": phi_int[:, j] for j, f in enumerate(feats)}, "base": ex1["base_stack"]}).to_parquet(int_path, index=False)
if "value" in args.parts:
    ex2 = explain_stack(models, w, b, Xl, bg2)
    Gp, Gi, Gi2, Gl = groups_of(phi_path), groups_of(phi_int), groups_of(ex2["phi_stack"]), groups_of(W_lime)
    out["value"] = {
        "additivity_error_interventional": err, "base_interventional": ex1["base_stack"],
        "feature": {"path_vs_int": compare(phi_path, phi_int, K), "int_vs_int_other_bg": compare(phi_int, ex2["phi_stack"], K),
                    "lime_vs_path": compare(W_lime, phi_path, K), "lime_vs_int": compare(W_lime, phi_int, K)},
        "group": {"path_vs_int": compare(Gp, Gi, 1), "int_vs_int_other_bg": compare(Gi, Gi2, 1),
                  "lime_vs_path": compare(Gl, Gp, 1), "lime_vs_int": compare(Gl, Gi, 1)},
        "global_group_importance": {"path": global_importance(Gp).tolist(), "interventional": global_importance(Gi).tolist(),
                                    "lime": global_importance(Gl).tolist()},
        "global_feature_importance": {"path": global_importance(phi_path).tolist(), "interventional": global_importance(phi_int).tolist()},
    }
    v = out["value"]
    log.info("path vs interventional: Spearman từng luồng %.3f, τ nhóm toàn cục %.3f | hai tập nền: %.3f",
             v["feature"]["path_vs_int"]["spearman_flow_mean"], v["group"]["path_vs_int"]["global_kendall"],
             v["feature"]["int_vs_int_other_bg"]["spearman_flow_mean"])
    save()
if phi_int is None and int_path.exists():
    ip = pd.read_parquet(int_path)
    if np.array_equal(ip["test_idx"].to_numpy(), lime_idx):
        phi_int = ip[[f"int_{f}" for f in feats]].to_numpy(dtype="float64")

score_fn = make_proba_fn(models, w, b, feats)
A_rows = Xl.to_numpy(dtype="float64")

# ── knn: phép xóa trong phân phối ───────────────────────────────────────────
if "knn" in args.parts:
    t0 = time.time()
    ref_idx = np.random.default_rng(seed + 12).choice(ben_idx, min(args.knn_ref, len(ben_idx)), replace=False)
    imp = NeighborImputer(Xtr.iloc[ref_idx].to_numpy(dtype="float64"))
    all_idx = np.random.default_rng(seed + 13).choice(len(Xtr), min(args.knn_ref, len(Xtr)), replace=False)
    ood_ref = NeighborImputer(Xtr.iloc[all_idx].to_numpy(dtype="float64"))
    attr = {"shap_path": phi_path, "shap_int": phi_int, "lime": W_lime}
    kks = sorted(set(args.knn_ks))
    out["knn"] = {
        "median_fill": deletion_test(score_fn, A_rows, attr, baseline, kks, thr, np.random.default_rng(seed + 3)),
        "knn_fill": knn_deletion_test(score_fn, A_rows, attr, imp, baseline, kks, thr, np.random.default_rng(seed + 3),
                                      ood_ref=ood_ref),
        "n_neighbors": 5, "ref_size": int(len(ref_idx)),
    }
    kk = "5" if 5 in kks else str(max(kks))
    log.info("Xóa %s đặc trưng, thay bằng láng giềng lành tính (%.0fs): mất cảnh báo SHAP path %.0f%%, SHAP int %.0f%%, LIME %.0f%%, ngẫu nhiên %.0f%%",
             kk, time.time() - t0, *[100 * out["knn"]["knn_fill"][kk][m]["flip_rate"] for m in ("shap_path", "shap_int", "lime", "random")])
    save()
    # xóa theo nhóm, trên cùng các luồng của LIME
    gnames = group_attribution(phi_path, feats, groups, GROUP_ORDER)[1]
    gcols = [[j for j, f in enumerate(feats) if groups[f] == g] for g in gnames]
    gattr = {"shap_path": groups_of(phi_path), "shap_int": groups_of(phi_int), "lime": groups_of(W_lime)}
    out["knn"]["group"] = knn_group_deletion_test(score_fn, A_rows, gattr, gcols, imp, args.knn_groups, thr,
                                                  np.random.default_rng(seed + 4))
    out["knn"]["group_names"] = gnames
    g1 = out["knn"]["group"][str(min(args.knn_groups))]
    log.info("Xóa %d nhóm: mất cảnh báo SHAP path %.0f%%, LIME %.0f%%, nhóm ngẫu nhiên %.0f%%, cột ngẫu nhiên cùng số lượng %.0f%%",
             min(args.knn_groups), *[100 * g1[m]["flip_rate"] for m in ("shap_path", "lime", "random_groups", "random_features")])
    save()
    # toàn bộ mẫu giải thích của bước 08: SHAP path so với ngẫu nhiên, mức đặc trưng và mức nhóm
    t1 = time.time()
    e_idx = es["test_idx"].to_numpy(dtype=int)
    XE = X.iloc[e_idx]
    exE = explain_stack(models, w, b, XE)
    GE = groups_of(exE["phi_stack"])
    GE_saved = es[[f"phi_{g}" for g in gnames]].to_numpy(dtype="float64")
    assert np.abs(GE - GE_saved).max() < 1e-6, "explain_sample.parquet không khớp mô hình hiện tại: chạy lại bước 08"
    XE_rows = XE.to_numpy(dtype="float64")
    out["knn"]["full"] = {
        "n_flows": int(len(e_idx)),
        "knn_fill": knn_deletion_test(score_fn, XE_rows, {"shap_path": exE["phi_stack"]}, imp, baseline, kks, thr,
                                      np.random.default_rng(seed + 5)),
        "group": knn_group_deletion_test(score_fn, XE_rows, {"shap_path": GE}, gcols, imp, args.knn_groups, thr,
                                         np.random.default_rng(seed + 6)),
    }
    fu = out["knn"]["full"]
    log.info("Toàn mẫu %d luồng (%.0fs): xóa %s đặc trưng mất cảnh báo %.0f%% (ngẫu nhiên %.0f%%); xóa %d nhóm %.0f%% (cột ngẫu nhiên %.0f%%)",
             len(e_idx), time.time() - t1, kk, 100 * fu["knn_fill"][kk]["shap_path"]["flip_rate"], 100 * fu["knn_fill"][kk]["random"]["flip_rate"],
             min(args.knn_groups), 100 * fu["group"][str(min(args.knn_groups))]["shap_path"]["flip_rate"],
             100 * fu["group"][str(min(args.knn_groups))]["random_features"]["flip_rate"])
    save()

# ── robust: độ bền cục bộ ───────────────────────────────────────────────────
if "robust" in args.parts:
    t0 = time.time()
    sub = stratified_pick(np.arange(len(lime_idx)), att[lime_idx], min(args.n_robust, len(lime_idx)), np.random.default_rng(seed + 30))
    Xs, z0 = A_rows[sub], z_all[lime_idx][sub]
    res = {"n_flows": int(len(sub)), "reps": args.robust_reps, "levels": {}}
    for rel in args.robust_rel:
        rows = {"shap_path": [], "shap_int": [], "dz": [], "still_flagged": []}
        for r in range(args.robust_reps):
            Xp = perturb(Xs, rel, np.random.default_rng(seed + 1000 + r + int(rel * 1e4)), bin_cols)
            Xp_df = pd.DataFrame(Xp, columns=feats)
            exp_p = explain_stack(models, w, b, Xp_df)
            exi_p = explain_stack(models, w, b, Xp_df, bg1)
            rows["shap_path"].append(compare(phi_path[sub], exp_p["phi_stack"], K))
            rows["shap_int"].append(compare(phi_int[sub], exi_p["phi_stack"], K))
            rows["shap_path"][-1]["group_top1_same"] = float((np.argmax(np.abs(groups_of(phi_path[sub])), 1) == np.argmax(np.abs(groups_of(exp_p["phi_stack"])), 1)).mean())
            rows["dz"].append(float(np.median(np.abs(exp_p["z"] - z0))))
            rows["still_flagged"].append(float((1 / (1 + np.exp(-exp_p["z"])) >= thr).mean()))
        lvl = {m: {key: float(np.mean([d[key] for d in rows[m]])) for key in rows[m][0] if key != "k"} for m in ("shap_path", "shap_int")}
        lvl.update({"median_abs_dz": float(np.mean(rows["dz"])), "still_flagged": float(np.mean(rows["still_flagged"]))})
        res["levels"][str(rel)] = lvl
        log.info("Nhiễu ±%.0f%%: SHAP path Spearman %.3f, top-5 Jaccard %.3f | |Δz| trung vị %.3f",
                 100 * rel, lvl["shap_path"]["spearman_flow_mean"], lvl["shap_path"]["topk_jaccard_mean"], lvl["median_abs_dz"])
    # LIME ở mức nhiễu nhỏ nhất: nhiễu đầu vào (seed khác) và chỉ seed (đầu vào giữ nguyên)
    rel = min(args.robust_rel)
    predict_fn = make_lime_predict_fn(models, w, b, feats, lc["target"])
    lin, lseed = [], []
    for r in range(args.robust_reps):
        Xp = perturb(Xs, rel, np.random.default_rng(seed + 1000 + r + int(rel * 1e4)), bin_cols)
        e1 = build_explainer(Xbg_lime, lc["target"], lc["discretizer"], lc["kernel_width"], seed + 200 + r)
        lin.append(compare(W_lime[sub], lime_weights(e1, predict_fn, Xp, lc["target"], lc["num_perturbations"])["weights"], K))
        e2 = build_explainer(Xbg_lime, lc["target"], lc["discretizer"], lc["kernel_width"], seed + 300 + r)
        lseed.append(compare(W_lime[sub], lime_weights(e2, predict_fn, Xs, lc["target"], lc["num_perturbations"])["weights"], K))
        log.info("  LIME lần %d/%d xong", r + 1, args.robust_reps)
    avg = lambda L: {key: float(np.mean([d[key] for d in L])) for key in L[0] if key != "k"}  # noqa: E731
    res["lime"] = {"rel": rel, "input_and_seed": avg(lin), "seed_only": avg(lseed)}
    res["seconds"] = round(time.time() - t0, 1)
    out["robust"] = res
    log.info("LIME (±%.0f%%): Spearman %.3f khi đổi đầu vào, %.3f khi chỉ đổi seed", 100 * rel,
             res["lime"]["input_and_seed"]["spearman_flow_mean"], res["lime"]["seed_only"]["spearman_flow_mean"])
    save()

# ── seeds: độ ổn định khi huấn luyện lại ────────────────────────────────────
if "seeds" in args.parts:
    t0 = time.time()
    exp_idx = es["test_idx"].to_numpy(dtype=int)
    common = pick_common(exp_idx, att, args.n_seed_flows, seed)
    Xc = X.iloc[common]
    dirs = [root / "models" / ds] + sorted(p for p in (root / "models").glob(f"{ds}_seed*") if (p / "stack.joblib").exists())
    phis, phis_int, G, Gint, info = [], [], [], [], []
    for d in dirs:
        m_k, w_k, b_k, thr_k = load_system(d)
        ex = explain_stack(m_k, w_k, b_k, Xc)
        exi = explain_stack(m_k, w_k, b_k, Xc, bg1)
        phis.append(ex["phi_stack"]); phis_int.append(exi["phi_stack"])
        G.append(groups_of(ex["phi_stack"])); Gint.append(groups_of(exi["phi_stack"]))
        info.append({"model": d.name, "meta_weights": dict(zip(BASE_NAMES, map(float, w_k))),
                     "flagged_share": float((1 / (1 + np.exp(-ex["z"])) >= thr_k).mean()),
                     "global_group_importance": global_importance(G[-1]).tolist()})
        log.info("  %s xong (%.0fs)", d.name, time.time() - t0)
    out["seeds"] = {"n_flows": int(len(common)), "models": info,
                    "path": {"feature": pairwise(phis, K), "group": pairwise(G, 1)},
                    "interventional": {"feature": pairwise(phis_int, K), "group": pairwise(Gint, 1)},
                    "seconds": round(time.time() - t0, 1)}
    s = out["seeds"]["path"]
    log.info("Qua %d mô hình: τ nhóm toàn cục trung bình %.3f (nhỏ nhất %.3f), Spearman từng luồng %.3f, top-5 Jaccard %.3f",
             len(dirs), s["group"]["global_kendall_mean"], s["group"]["global_kendall_min"],
             s["feature"]["spearman_flow_mean_mean"], s["feature"]["topk_jaccard_mean_mean"])
    save()

# ── weighted: thứ hạng toàn cục có trọng số ─────────────────────────────────
if "weighted" in args.parts:
    Ges = es[[f"phi_{g}" for g in GROUP_ORDER if f"phi_{g}" in es.columns]].to_numpy(dtype="float64")
    gn = [g for g in GROUP_ORDER if f"phi_{g}" in es.columns]
    pop = pd.Series(att[flagged]).astype(str).value_counts().to_dict()
    wts = stratum_weights(es[ATTACK_COL].astype(str).to_numpy(), es[WEIGHT_COL].to_numpy(), pop)
    imp_u, imp_w = global_importance(Ges), global_importance(Ges, wts)
    out["weighted"] = {"groups": gn, "unweighted": imp_u.tolist(), "weighted": imp_w.tolist(),
                       "kendall": kendall(imp_u, imp_w),
                       "rank_unweighted": [gn[i] for i in np.argsort(-imp_u)], "rank_weighted": [gn[i] for i in np.argsort(-imp_w)],
                       "attack_share_sample": es[ATTACK_COL].astype(str).value_counts(normalize=True).round(4).to_dict(),
                       "attack_share_alerts_weighted": (pd.Series(wts).groupby(es[ATTACK_COL].astype(str).to_numpy()).sum() / wts.sum()).round(4).to_dict()}
    log.info("Thứ hạng nhóm không trọng số: %s | có trọng số: %s | τ = %.3f",
             out["weighted"]["rank_unweighted"], out["weighted"]["rank_weighted"], out["weighted"]["kendall"])
    save()

out["seconds_total"] = round(time.time() - t_all, 1)
save()
log.info("Đã ghi reports/explain_checks_%s.json (%.0fs)", ds, time.time() - t_all)
