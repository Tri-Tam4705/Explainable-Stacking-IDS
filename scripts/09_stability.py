"""Bước 6: đánh giá giải thích SHAP gộp — ổn định, fidelity, so sánh với baseline E1/E2.

Ổn định (xem src/xsids/explain/stability.py):
  - Kiểm tra tính lại: tính lại SHAP gộp trên đúng các luồng của bước 5 và so thứ hạng nhóm TỪNG LUỒNG với bản đã lưu.
    TreeSHAP tất định nên τ phải bằng 1; khác 1 nghĩa là mô hình hoặc mẫu đã đổi sau bước 5 (phải chạy lại bước 08).
  - Bootstrap: lấy mẫu lại tập luồng được giải thích, đo thứ hạng TOÀN CỤC (mean |φ|) của nhóm và của đặc trưng đổi bao nhiêu.
Fidelity: R² của hồi quy tuyến tính φ_nhóm → z (gần như tất yếu nhờ tính cộng; chỉ để đối chiếu với E1).
E1: SHAP của một mô hình đơn (lgbm) — baseline "giải thích đơn".
E2: SHAP gộp không trọng số (trung bình đơn giản) — baseline "gộp thô".

Bản trước của script này so thứ hạng của luồng i với luồng bootstrap thứ i (một luồng KHÁC, chọn ngẫu nhiên), nên
τ ≈ 0,67 khi đó đo mức giống nhau giữa hai luồng bất kỳ chứ không phải độ ổn định, và tốn 20 lần tính SHAP (vài giờ).
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

import _bootstrap  # noqa: F401
from xsids.config import load_config, resolve_path
from xsids.explain.lime_stack import recover_sample_idx
from xsids.explain.shap_stack import agreement_score, explain_stack, group_attribution
from xsids.explain.stability import bootstrap_rank_stability, kendall_tau_rows
from xsids.features.engineer import GROUP_ORDER
from xsids.models.data import load_feature_set, load_split, xyw
from xsids.models.stacking import raw_matrix, stack_logit
from xsids.models.zoo import BASE_NAMES, load_models_cfg
from xsids.report.summaries import load_json, stability_text
from xsids.schema import ATTACK_COL
from xsids.utils import get_logger, write_json

log = get_logger()
ap = argparse.ArgumentParser()
ap.add_argument("--dataset", choices=["unsw", "cse"], required=True)
ap.add_argument("--n-bootstrap", type=int, default=200, help="số lần bootstrap tập luồng (rẻ: không tính lại SHAP)")
ap.add_argument("--top-k", type=int, default=5, help="k cho Jaccard top-k đặc trưng giữa các lần bootstrap")
ap.add_argument("--processed-dir", default=None)
ap.add_argument("--models-dir", default=None)
args = ap.parse_args()

cfg, mcfg = load_config(), load_models_cfg()
proc = Path(args.processed_dir) if args.processed_dir else resolve_path(cfg, "processed_dir")
mdir = Path(args.models_dir) if args.models_dir else Path(cfg["_root"]) / "models" / args.dataset
rdir = resolve_path(cfg, "reports_dir")

fs = load_feature_set(proc)
feats, groups = fs["features"], fs["groups"]
models = {n: joblib.load(mdir / f"{n}.joblib") for n in BASE_NAMES}
st = joblib.load(mdir / "stack.joblib")
w, b = st["w"], st["b"]

# ── mẫu luồng của bước 5 ─────────────────────────────────────────────────────
sample = pd.read_parquet(proc / args.dataset / "explain_sample.parquet")
gnames = [g for g in GROUP_ORDER if f"phi_{g}" in sample.columns]
G_saved = np.column_stack([sample[f"phi_{g}"].to_numpy() for g in gnames])
z_saved = sample["z"].to_numpy()

te = load_split(proc, args.dataset, "test")
X_te, _, _, att_te = xyw(te, feats)
if "test_idx" in sample.columns:
    pick = sample["test_idx"].to_numpy(dtype=int)
else:  # file của bản cũ: ghép lại theo (tên tấn công, logit z)
    z_all = stack_logit(raw_matrix(models, X_te), w, b)
    pick = recover_sample_idx(z_all, att_te, z_saved, sample[ATTACK_COL].astype(str).to_numpy())
    if pick is None:
        raise SystemExit("Không ghép lại được mẫu của bước 5 với tập test. Chạy lại: python scripts/08_explain.py "
                         f"--dataset {args.dataset}")
Xs = X_te.iloc[pick]
n_sample = len(Xs)

# ── 1. tính lại SHAP gộp một lần: kiểm tra tất định và lấy φ mức đặc trưng ──
log.info("Tính lại SHAP gộp cho %d luồng (một lần)...", n_sample)
ex = explain_stack(models, w, b, Xs)
phi = ex["phi_stack"]
G_new, gnames_new = group_attribution(phi, feats, groups, GROUP_ORDER)
if gnames_new != gnames:
    raise SystemExit(f"Nhóm trong explain_sample.parquet {gnames} khác feature_set.json {gnames_new}; chạy lại bước 08.")
tau_flow = kendall_tau_rows(np.abs(G_saved), np.abs(G_new))
recompute = {
    "n_flows": int(n_sample),
    "per_flow_kendall_tau_mean": float(np.nanmean(tau_flow)),
    "per_flow_kendall_tau_min": float(np.nanmin(tau_flow)),
    "max_abs_diff_group_phi": float(np.abs(G_saved - G_new).max()),
    "max_abs_diff_z": float(np.abs(z_saved - ex["z"]).max()),
}
recompute["matches_saved"] = bool(recompute["max_abs_diff_group_phi"] < 1e-3 and recompute["max_abs_diff_z"] < 1e-3)
log.info("Kiểm tra tính lại: τ từng luồng TB %.4f (nhỏ nhất %.4f), lệch |φ nhóm| tối đa %.2e → %s",
         recompute["per_flow_kendall_tau_mean"], recompute["per_flow_kendall_tau_min"], recompute["max_abs_diff_group_phi"],
         "KHỚP bản đã lưu" if recompute["matches_saved"] else "KHÔNG khớp (mô hình hoặc mẫu đã đổi sau bước 08)")

# ── 2. bootstrap tập luồng: thứ hạng toàn cục của nhóm và của đặc trưng ─────
rng = np.random.default_rng(mcfg["seed"])
stab_group = bootstrap_rank_stability(G_new, args.n_bootstrap, rng)
stab_feat = bootstrap_rank_stability(phi, args.n_bootstrap, rng, top_k=args.top_k)
stab_group["level"] = "global_group_ranking"
imp_g = np.abs(G_new).mean(axis=0)
stab_group["group_ranking"] = [gnames[i] for i in np.argsort(-imp_g)]
imp_f = np.abs(phi).mean(axis=0)
stab_feat["top_features"] = [feats[i] for i in np.argsort(-imp_f)[:args.top_k]]
del stab_feat["per_bootstrap"]
log.info("Ổn định thứ hạng toàn cục (%d bootstrap): nhóm τ = %.4f ± %.4f | đặc trưng τ = %.4f ± %.4f, Jaccard top-%d = %.3f",
         args.n_bootstrap, stab_group["kendall_tau_mean"], stab_group["kendall_tau_std"],
         stab_feat["kendall_tau_mean"], stab_feat["kendall_tau_std"], args.top_k, stab_feat["topk_jaccard_mean"])

# ── 3. fidelity: R² hồi quy tuyến tính φ_nhóm → z ───────────────────────────
fidelity_r2 = float(LinearRegression().fit(G_new, ex["z"]).score(G_new, ex["z"]))

# ── 4. E1 (LightGBM đơn) và E2 (gộp đều) so với SHAP gộp theo trọng số ──────
G_base = [group_attribution(ph, feats, groups, GROUP_ORDER)[0] for ph in ex["phi_per_base"]]
G_lgbm = G_base[BASE_NAMES.index("lgbm")]
A_e1 = agreement_score([G_lgbm, G_new])
A_e2 = agreement_score([np.mean(G_base, axis=0), G_new])
fidelity_e1 = float(LinearRegression().fit(G_lgbm, ex["z"]).score(G_lgbm, ex["z"]))

result = {
    "dataset": args.dataset,
    "n_sample": n_sample,
    "n_bootstrap": args.n_bootstrap,
    "stability": stab_group,
    "stability_feature": stab_feat,
    "recompute_check": recompute,
    "fidelity": {"r2_group_to_z": fidelity_r2},
    "e1_lgbm_single": {
        "agreement_with_stack_mean": float(np.nanmean(A_e1)),
        "agreement_with_stack_p50": float(np.nanpercentile(A_e1, 50)),
        "fidelity_r2_group_to_z": fidelity_e1,
    },
    "e2_uniform_ensemble": {
        "agreement_with_stack_mean": float(np.nanmean(A_e2)),
        "agreement_with_stack_p50": float(np.nanpercentile(A_e2, 50)),
    },
}
write_json(result, rdir / f"stability_{args.dataset}.json")
log.info("Đã ghi reports/stability_%s.json", args.dataset)

# In tóm tắt (cùng định dạng với scripts/13_summary.py)
print(stability_text(args.dataset, result, load_json(rdir / f"explain_{args.dataset}.json")))
