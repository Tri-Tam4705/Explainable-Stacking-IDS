"""Shapley theo nhóm đặc trưng với phép xóa trong phân phối: độ trung thực so với mức tối đa (xem
src/xsids/explain/group_shapley.py).

Ba phần:
  group  : trên mẫu con cố định của mẫu giải thích (giống bước 22, xem xsids.explain.common.explain_subsample). Bảng
           2^6 liên minh nhóm với ba tập tham chiếu lành tính rời nhau của train: A (k = 5 láng giềng, đúng phép đo của
           bước 22) để ĐÁNH GIÁ; B (k = 25) để TÍNH Shapley nhóm; C (k = 25) để đo độ ổn định khi đổi tập tham chiếu.
           So SHAP gộp path-dependent, SHAP gộp interventional và Shapley nhóm theo phép xóa g nhóm (kèm oracle và kỳ
           vọng ngẫu nhiên) và tương quan trung thực. Kiểm tra chéo: hồi quy kernel Shapley trên đủ 64 liên minh phải
           cho lại đúng Shapley nhóm.
  full   : toàn bộ mẫu giải thích của bước 08: SHAP gộp path-dependent so với Shapley nhóm, cùng phép đo.
  feature: mức đặc trưng trên n luồng: KernelSHAP lấy mẫu liên minh với phép xóa interventional (cùng tập nền 100 luồng
           của SHAP gộp interventional), so với SHAP gộp interventional chính xác, kèm thời gian trên mỗi luồng.

Dùng:
    python scripts/24_group_shapley.py --dataset unsw
    python scripts/24_group_shapley.py --dataset cse --parts group --fast
Đầu ra: reports/group_shapley_<ds>.json. Không ghi đè mô hình hay kết quả chính.
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
from xsids.explain.checks import NeighborImputer, compare
from xsids.explain.group_shapley import (
    coalition_table,
    deletion_eval,
    exact_shapley,
    faithfulness_correlation,
    fit_kernel_shap,
    interventional_values,
    kernel_attribution,
    sample_masks,
    shapley_kernel,
)
from xsids.explain.common import explain_subsample, make_logit_fn, rank_corr_rows
from xsids.explain.shap_stack import INTERVENTIONAL_BG_MAX, explain_stack, group_attribution
from xsids.features.engineer import GROUP_ORDER
from xsids.models.data import load_feature_set, load_split, xyw
from xsids.models.zoo import BASE_NAMES, load_models_cfg
from xsids.utils import get_logger, write_json

PARTS = ["group", "full", "feature"]
log = get_logger()
ap = argparse.ArgumentParser()
ap.add_argument("--dataset", choices=["unsw", "cse"], required=True)
ap.add_argument("--parts", nargs="+", choices=PARTS, default=PARTS)
ap.add_argument("--ref", type=int, default=20000, help="số luồng lành tính của train trong mỗi tập tham chiếu")
ap.add_argument("--k-eval", type=int, default=5, help="số láng giềng của phép đo (giống bước 22)")
ap.add_argument("--k-explain", type=int, default=25, help="số láng giềng khi tính Shapley nhóm")
ap.add_argument("--gs", type=int, nargs="+", default=[1, 2, 3], help="số nhóm bị xóa")
ap.add_argument("--n-feature", type=int, default=100, help="số luồng cho phần mức đặc trưng")
ap.add_argument("--n-pairs", type=int, default=512, help="số cặp liên minh lấy mẫu mỗi luồng (mức đặc trưng)")
ap.add_argument("--fast", action="store_true", help="tham số nhỏ để chạy thử")
args = ap.parse_args()
if args.fast:
    args.ref, args.n_feature, args.n_pairs = 2000, 10, 64

cfg, mcfg = load_config(), load_models_cfg()
ds, seed = args.dataset, mcfg["seed"]
ecfg = dict(mcfg["explain"])
if args.fast:
    ecfg.update(mcfg.get("fast_explain") or {})
root, proc, rdir = Path(cfg["_root"]), resolve_path(cfg, "processed_dir"), resolve_path(cfg, "reports_dir")
fs = load_feature_set(proc)
feats, groups = fs["features"], fs["groups"]

st = joblib.load(root / "models" / ds / "stack.joblib")
models = {n: joblib.load(root / "models" / ds / f"{n}.joblib") for n in BASE_NAMES}
w, b, thr = np.asarray(st["w"]), float(st["b"]), float(st["threshold"])
z_fn = make_logit_fn(models, w, b, feats)

te = load_split(proc, ds, "test")
X, y, wt, att = xyw(te, feats)
es0 = pd.read_parquet(proc / ds / "explain_sample.parquet")
sub_idx = explain_subsample(es0["test_idx"].to_numpy(dtype=int), att, ecfg["n_subsample"], seed)
XL = X.iloc[sub_idx]
assert np.allclose(es0.set_index("test_idx").loc[sub_idx, "z"].to_numpy(), z_fn(XL.to_numpy()), atol=1e-5), "explain_sample.parquet không khớp mô hình hiện tại: chạy lại bước 08"
phi_path = explain_stack(models, w, b, XL)["phi_stack"]  # SHAP gộp path-dependent, tất định, trùng bước 08

tr = load_split(proc, ds, "train")
Xtr, ytr, _, _ = xyw(tr, feats)
ben = np.flatnonzero(ytr == 0)
bg1 = Xtr.iloc[np.random.default_rng(seed + 10).choice(ben, min(500, len(ben)), replace=False)]  # như bước 22


def draw(exclude, s):
    pool = np.setdiff1d(ben, exclude)
    pool = pool if len(pool) >= args.ref else ben
    return np.random.default_rng(s).choice(pool, min(args.ref, len(pool)), replace=False)


refA = np.random.default_rng(seed + 12).choice(ben, min(args.ref, len(ben)), replace=False)  # đúng tập của bước 22
refB = draw(refA, seed + 31)
refC = draw(np.r_[refA, refB], seed + 32)
imps = {k: NeighborImputer(Xtr.iloc[v].to_numpy(dtype="float64")) for k, v in (("A", refA), ("B", refB), ("C", refC))}

gnames = group_attribution(phi_path, feats, groups, GROUP_ORDER)[1]
gcols = [[j for j, f in enumerate(feats) if groups[f] == g] for g in gnames]


def by_group(A):
    return group_attribution(np.asarray(A, dtype="float64"), feats, groups, GROUP_ORDER)[0]


def agree(A, B, k=1) -> dict:
    out = compare(A, B, k)
    out["spearman_signed_mean"] = float(np.nanmean(rank_corr_rows(np.asarray(A), np.asarray(B))))
    out["top1_signed_same"] = float((np.argmax(A, axis=1) == np.argmax(B, axis=1)).mean())
    return out


out_path = rdir / f"group_shapley_{ds}.json"
out = json.load(open(out_path, encoding="utf-8")) if out_path.exists() else {}
if out.get("fast", args.fast) != args.fast:
    out = {}
out.update({"dataset": ds, "fast": args.fast, "threshold": thr, "groups": gnames, "ref_size": int(len(refA)),
            "k_eval": args.k_eval, "k_explain": args.k_explain, "n_sub_flows": int(len(sub_idx))})


def save():
    write_json(out, out_path)


def tables(Xrows):
    t0 = time.time()
    tA = coalition_table(Xrows, z_fn, imps["A"], gcols, k=args.k_eval, seed=seed)
    tB = coalition_table(Xrows, z_fn, imps["B"], gcols, k=args.k_explain, seed=seed + 1)
    secs = (time.time() - t0) / len(Xrows)
    return tA, tB, secs


t_all = time.time()
if "group" in args.parts:
    ip = proc / ds / "explain_int_sample.parquet"
    if ip.exists() and np.array_equal(pd.read_parquet(ip)["test_idx"].to_numpy(), sub_idx):
        phi_int = pd.read_parquet(ip)[[f"int_{f}" for f in feats]].to_numpy(dtype="float64")
    else:
        phi_int = explain_stack(models, w, b, XL, bg1)["phi_stack"]
    rows = XL.to_numpy(dtype="float64")
    tA, tB, secs = tables(rows)
    t0 = time.time()
    tC = coalition_table(rows, z_fn, imps["C"], gcols, k=args.k_explain, seed=seed + 2)
    secs_one = (time.time() - t0) / len(rows)
    onm, onmC = exact_shapley(tB["Z"], tB["masks"]), exact_shapley(tC["Z"], tC["masks"])
    kernel_check = kernel_attribution(tB["Z"], tB["masks"], shapley_kernel(tB["masks"]))
    attrs = {"shap_path": by_group(phi_path), "shap_int": by_group(phi_int), "group_shapley": onm}
    ceiling = exact_shapley(tA["Z"], tA["masks"])  # Shapley nhóm tính bằng chính phép đo: mức trần của tương quan
    out["group"] = {
        "deletion": deletion_eval(tA["Z"], tA["P"], tA["masks"], attrs, thr, args.gs),
        "faithfulness_correlation": faithfulness_correlation(tA["Z"], tA["masks"], {**attrs, "ceiling_same_operator": ceiling}),
        "stability_ref_B_vs_C": agree(onm, onmC),
        "shap_vs_group_shapley": {"shap_path": agree(attrs["shap_path"], onm), "shap_int": agree(attrs["shap_int"], onm)},
        "kernel_check_max_abs_diff": float(np.abs(kernel_check - onm).max()),
        "seconds_per_flow_one_table": round(secs_one, 4),
        "global_importance": {k: np.abs(v).mean(0).tolist() for k, v in attrs.items()},
    }
    g = out["group"]
    log.info("Nhóm (%d luồng của mẫu con): xóa 1 nhóm mất cảnh báo SHAP path %.0f%%, Shapley nhóm %.0f%%, oracle %.0f%%, ngẫu nhiên %.0f%% | tương quan trung thực %.2f -> %.2f (trần %.2f)",
             len(rows), *[100 * g["deletion"]["1"][m]["flip_rate"] for m in ("shap_path", "group_shapley", "oracle", "random")],
             g["faithfulness_correlation"]["shap_path"]["mean"], g["faithfulness_correlation"]["group_shapley"]["mean"],
             g["faithfulness_correlation"]["ceiling_same_operator"]["mean"])
    log.info("Shapley nhóm: ổn định B so với C %.2f, trùng hạng 1 %.0f%% | kiểm tra chéo kernel Shapley: sai lệch lớn nhất %.1e",
             g["stability_ref_B_vs_C"]["spearman_signed_mean"], 100 * g["stability_ref_B_vs_C"]["top1_signed_same"],
             g["kernel_check_max_abs_diff"])
    save()

if "full" in args.parts:
    es = pd.read_parquet(proc / ds / "explain_sample.parquet")
    e_idx = es["test_idx"].to_numpy(dtype=int)
    GE = es[[f"phi_{g}" for g in gnames]].to_numpy(dtype="float64")
    rows = X.iloc[e_idx].to_numpy(dtype="float64")
    assert np.allclose(es["z"].to_numpy(), z_fn(rows), atol=1e-5), "explain_sample.parquet không khớp mô hình hiện tại"
    tA, tB, secs = tables(rows)
    onm = exact_shapley(tB["Z"], tB["masks"])
    attrs = {"shap_path": GE, "group_shapley": onm}
    out["full"] = {"n_flows": int(len(rows)),
                   "deletion": deletion_eval(tA["Z"], tA["P"], tA["masks"], attrs, thr, args.gs),
                   "faithfulness_correlation": faithfulness_correlation(tA["Z"], tA["masks"], {**attrs, "ceiling_same_operator": exact_shapley(tA["Z"], tA["masks"])}),
                   "shap_path_vs_group_shapley": agree(GE, onm),
                   "global_importance": {k: np.abs(v).mean(0).tolist() for k, v in attrs.items()}}
    f = out["full"]
    log.info("Toàn mẫu %d luồng: mất cảnh báo khi xóa 1/2 nhóm: SHAP path %.0f%%/%.0f%%, Shapley nhóm %.0f%%/%.0f%%, oracle %.0f%%/%.0f%%, ngẫu nhiên %.0f%%/%.0f%%",
             len(rows), *[100 * f["deletion"][gg][m]["flip_rate"] for m in ("shap_path", "group_shapley", "oracle", "random") for gg in ("1", "2")][::1])
    save()

if "feature" in args.parts:
    import shap

    rng = np.random.default_rng(seed + 33)
    pick = np.sort(rng.choice(len(sub_idx), min(args.n_feature, len(sub_idx)), replace=False))
    Xf = XL.iloc[pick]
    t0 = time.time()
    exact = explain_stack(models, w, b, Xf, bg1)["phi_stack"]
    sec_exact = (time.time() - t0) / len(Xf)
    B = shap.utils.sample(bg1, INTERVENTIONAL_BG_MAX, random_state=0).to_numpy(dtype="float64")  # đúng tập nền của TreeSHAP
    v_empty = float(z_fn(B).mean())
    kshap, secs = [], []
    for i, x in enumerate(Xf.to_numpy(dtype="float64")):
        t1 = time.time()
        masks = sample_masks(len(feats), args.n_pairs, np.random.default_rng(seed + 1000 + i))
        vals = interventional_values(x, masks, B, z_fn)
        v_full = float(z_fn(x[None, :])[0])
        kshap.append(fit_kernel_shap(masks, vals, v_full, v_empty))
        secs.append(time.time() - t1)
    kshap = np.array(kshap)
    out["feature"] = {
        "n_flows": int(len(Xf)), "n_coalitions": int(2 * args.n_pairs), "background_size": int(len(B)),
        "seconds_per_flow": {"shap_stack_exact_interventional": round(sec_exact, 4), "sampled_kernelshap": round(float(np.mean(secs)), 4)},
        "kernelshap_vs_exact_interventional": agree(kshap, exact, 5),
        "kernelshap_vs_shap_path": agree(kshap, phi_path[pick], 5),
        "kernelshap_abs_error": {"mean": float(np.abs(kshap - exact).mean()), "max": float(np.abs(kshap - exact).max()),
                                 "relative_to_mean_abs_phi": float(np.abs(kshap - exact).mean() / np.abs(exact).mean())},
    }
    fe = out["feature"]
    log.info("Mức đặc trưng (%d luồng, %d liên minh): KernelSHAP so với SHAP gộp interventional, Spearman %.2f | thời gian/luồng: chính xác %.3fs, KernelSHAP %.2fs",
             len(Xf), 2 * args.n_pairs, fe["kernelshap_vs_exact_interventional"]["spearman_signed_mean"],
             fe["seconds_per_flow"]["shap_stack_exact_interventional"], fe["seconds_per_flow"]["sampled_kernelshap"])
    save()

out["seconds_total"] = round(time.time() - t_all, 1)
save()
log.info("Đã ghi %s (%.0fs)", out_path, time.time() - t_all)
