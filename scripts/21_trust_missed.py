"""Hai kiểm tra bổ sung trên tập test, dùng mô hình chính (models/<ds>/, seed 42):

1. A(x) có báo trước cảnh báo sai không. Trong số luồng bị gắn cờ, lấy mẫu cảnh báo sai (FP) và cảnh báo đúng (TP), tính
   SHAP gộp và A(x), rồi đo ROC-AUC của (−A) khi tách FP khỏi TP. So với độ tự tin của mô hình (−z), và đo lại AUC của A(x)
   trong từng khoảng tứ phân vị của z để xem A(x) có thêm thông tin ngoài z không.
2. Vì sao tấn công bị bỏ sót. Lấy mẫu luồng tấn công không bị gắn cờ (FN), tính SHAP gộp và so đóng góp theo nhóm đặc trưng
   với các luồng tấn công bị gắn cờ (TP) cùng tên tấn công.

Trọng số: mẫu lấy đều theo vector duy nhất trong từng vai trò (FP, TP, FN), nên mỗi vector nhận trọng số
n_orig × (số vector của vai trò đó / số vector đã lấy) để khôi phục tỷ lệ của toàn bộ tập test.

Dùng:
    python scripts/21_trust_missed.py --dataset unsw
    python scripts/21_trust_missed.py --dataset cse --n 200   # mẫu nhỏ hơn (mặc định 400)
Đầu ra: reports/trust_<ds>.json và data/processed/<ds>/trust_sample.parquet. Không ghi đè kết quả chính.
Thời gian: TreeSHAP của Extra Trees chậm nhất, khoảng 0,5 đến 1 giây mỗi luồng.
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import _bootstrap  # noqa: F401
from xsids.config import load_config, resolve_path
from xsids.explain.shap_stack import agreement_score, explain_stack, group_attribution
from xsids.explain.trust import bootstrap_ci, quantile_bins, rate_by_bins, stratified_auc, weighted_auc
from xsids.features.engineer import GROUP_ORDER
from xsids.models.data import load_feature_set, load_split, xyw
from xsids.models.stacking import raw_matrix, stack_logit
from xsids.models.zoo import BASE_NAMES, load_models_cfg
from xsids.schema import ATTACK_COL, FAMILY_COL
from xsids.utils import get_logger, write_json

log = get_logger()
ap = argparse.ArgumentParser()
ap.add_argument("--dataset", choices=["unsw", "cse"], required=True)
ap.add_argument("--n", type=int, default=400, help="số vector tối đa lấy cho mỗi vai trò (FP, TP, FN)")
ap.add_argument("--n-boot", type=int, default=500)
args = ap.parse_args()

cfg, mcfg = load_config(), load_models_cfg()
ds, seed = args.dataset, mcfg["seed"]
proc, rdir = resolve_path(cfg, "processed_dir"), resolve_path(cfg, "reports_dir")
mdir = Path(cfg["_root"]) / "models" / ds
fs = load_feature_set(proc)
feats, groups = fs["features"], fs["groups"]
models = {n: joblib.load(mdir / f"{n}.joblib") for n in BASE_NAMES}
st = joblib.load(mdir / "stack.joblib")
w, b, thr = st["w"], st["b"], st["threshold"]

te = load_split(proc, ds, "test")
X, y, wt, att = xyw(te, feats)
z = stack_logit(raw_matrix(models, X), w, b)
flag = 1 / (1 + np.exp(-z)) >= thr
role_all = np.where(flag & (y == 0), "FP", np.where(flag & (y == 1), "TP", np.where(~flag & (y == 1), "FN", "TN")))
rng = np.random.default_rng(seed)
pick, role, inv = [], [], []
for r in ("FP", "TP", "FN"):
    idx = np.flatnonzero(role_all == r)
    k = min(args.n, len(idx))
    if k == 0:
        continue
    pick.append(np.sort(rng.choice(idx, size=k, replace=False)))
    role += [r] * k
    inv += [len(idx) / k] * k
    log.info("%s: %d vector trong tập test, lấy %d", r, len(idx), k)
pick, role, inv = np.concatenate(pick), np.array(role), np.array(inv)

ex = explain_stack(models, w, b, X.iloc[pick])
G, gnames = group_attribution(ex["phi_stack"], feats, groups, GROUP_ORDER)
G_base = [group_attribution(ph, feats, groups, GROUP_ORDER)[0] for ph in ex["phi_per_base"]]
A = agreement_score(G_base)
err = float(ex["additivity_error"]["stack"].max())
log.info("Tính cộng của mẫu: sai số tối đa %.1e", err)

sub = te.iloc[pick].reset_index(drop=True)
df = pd.DataFrame(G, columns=[f"phi_{g}" for g in gnames])
df["z"], df["agreement"], df["role"] = ex["z"], A, role
df["w_pop"] = wt[pick] * inv
df[ATTACK_COL], df[FAMILY_COL] = sub[ATTACK_COL].to_numpy(), sub[FAMILY_COL].to_numpy()
df["test_idx"] = pick
df.to_parquet(proc / ds / "trust_sample.parquet", index=False)

# ── 1. A(x) và cảnh báo sai ────────────────────────────────────────────────────
fl = df[df.role.isin(["FP", "TP"])].reset_index(drop=True)
yfp = (fl.role == "FP").to_numpy().astype(int)
Af, zf, wf = fl.agreement.to_numpy(), fl.z.to_numpy(), fl.w_pop.to_numpy()
zbin = quantile_bins(zf, 4)
abin = quantile_bins(Af, 4)
trust = {}
if yfp.sum() > 0 and (1 - yfp).sum() > 0:
    n = len(fl)
    trust = {
        "n_fp": int(yfp.sum()), "n_tp": int((1 - yfp).sum()),
        "auc_low_agreement_detects_fp": weighted_auc(-Af, yfp, wf),
        "auc_low_agreement_ci95": bootstrap_ci(lambda i: weighted_auc(-Af[i], yfp[i], wf[i]), n, args.n_boot, seed),
        "auc_low_z_detects_fp": weighted_auc(-zf, yfp, wf),
        "auc_low_z_ci95": bootstrap_ci(lambda i: weighted_auc(-zf[i], yfp[i], wf[i]), n, args.n_boot, seed),
        "auc_low_agreement_within_z_quartiles": stratified_auc(-Af, yfp, zbin, wf),
        "auc_within_z_ci95": bootstrap_ci(lambda i: stratified_auc(-Af[i], yfp[i], zbin[i], wf[i]), n, args.n_boot, seed),
        "fp_share_by_agreement_quartile": rate_by_bins(abin, yfp, wf, 4),
        "agreement_quartile_edges": [float(v) for v in np.nanquantile(Af, [0.25, 0.5, 0.75])],
        "spearman_agreement_z": float(spearmanr(Af, zf, nan_policy="omit").statistic),
        "agreement_mean": {"FP": float(np.nanmean(Af[yfp == 1])), "TP": float(np.nanmean(Af[yfp == 0]))},
        "agreement_median": {"FP": float(np.nanmedian(Af[yfp == 1])), "TP": float(np.nanmedian(Af[yfp == 0]))},
    }
    log.info("A(x) → FP: AUC %.3f %s | z → FP: AUC %.3f | A(x) trong từng khoảng z: AUC %.3f",
             trust["auc_low_agreement_detects_fp"], trust["auc_low_agreement_ci95"], trust["auc_low_z_detects_fp"],
             trust["auc_low_agreement_within_z_quartiles"])

# ── 2. tấn công bị bỏ sót ───────────────────────────────────────────────────────
z_thr = float(np.log(thr / (1 - thr)))
att_df = df[df.role.isin(["FN", "TP"])]
missed = {"z_threshold": z_thr, "groups": gnames, "by_attack": {}}
for fam, g in att_df.groupby(ATTACK_COL, observed=True):
    fn, tp = g[g.role == "FN"], g[g.role == "TP"]
    if len(fn) < 10:
        continue
    def wmean(d):
        return {c: float(np.average(d[f"phi_{c}"], weights=d.w_pop)) for c in gnames} if len(d) else None
    missed["by_attack"][str(fam)] = {
        "n_fn": int(len(fn)), "n_tp": int(len(tp)),
        "fn_weight_share": float(fn.w_pop.sum() / df[df.role == "FN"].w_pop.sum()),
        "mean_phi_fn": wmean(fn), "mean_phi_tp": wmean(tp),
        "mean_z_fn": float(np.average(fn.z, weights=fn.w_pop)),
        "mean_z_tp": float(np.average(tp.z, weights=tp.w_pop)) if len(tp) else None,
    }
fn_all = df[df.role == "FN"]
if len(fn_all):
    missed["all_fn"] = {"n": int(len(fn_all)), "mean_z": float(np.average(fn_all.z, weights=fn_all.w_pop)),
                        "mean_phi": {c: float(np.average(fn_all[f"phi_{c}"], weights=fn_all.w_pop)) for c in gnames}}

out = {"dataset": ds, "threshold": thr, "n_per_role": {r: int((role == r).sum()) for r in ("FP", "TP", "FN")},
       "population_per_role": {r: int((role_all == r).sum()) for r in ("FP", "TP", "FN", "TN")},
       "additivity_error_max": err, "trust": trust, "missed": missed}
write_json(out, rdir / f"trust_{ds}.json")
log.info("Đã ghi reports/trust_%s.json", ds)
for fam, v in missed["by_attack"].items():
    top = sorted(v["mean_phi_fn"].items(), key=lambda t: t[1])[:2]
    log.info("  %s: %d FN, z trung bình %.2f (ngưỡng %.2f); nhóm kéo về lành tính mạnh nhất: %s", fam, v["n_fn"],
             v["mean_z_fn"], z_thr, ", ".join(f"{g} {x:+.2f}" for g, x in top))
