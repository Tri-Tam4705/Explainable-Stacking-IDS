"""Bước 5: giải thích các luồng bị gắn cờ bằng TreeSHAP gộp theo trọng số meta-learner."""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from xsids.config import load_config, resolve_path
from xsids.explain.shap_stack import agreement_score, explain_stack, group_attribution
from xsids.features.engineer import GROUP_ORDER
from xsids.models.data import load_feature_set, load_split, xyw
from xsids.models.stacking import raw_matrix, stack_logit
from xsids.models.zoo import BASE_NAMES, load_models_cfg
from xsids.schema import ATTACK_COL, FAMILY_COL, LABEL_COL, WEIGHT_COL
from xsids.report.summaries import explain_text
from xsids.utils import get_logger, write_json

log = get_logger()
ap = argparse.ArgumentParser()
ap.add_argument("--dataset", choices=["unsw", "cse"], required=True)
ap.add_argument("--n", type=int, default=None, help="số luồng bị gắn cờ cần giải thích")
ap.add_argument("--processed-dir", default=None)
ap.add_argument("--models-dir", default=None)
ap.add_argument("--tag", default="")
args = ap.parse_args()

cfg, mcfg = load_config(), load_models_cfg()
proc = Path(args.processed_dir) if args.processed_dir else resolve_path(cfg, "processed_dir")
mdir = Path(args.models_dir) if args.models_dir else Path(cfg["_root"]) / "models" / args.dataset
fs = load_feature_set(proc)
feats, groups = fs["features"], fs["groups"]
models = {n: joblib.load(mdir / f"{n}.joblib") for n in BASE_NAMES}
st = joblib.load(mdir / "stack.joblib")
w, b, thr = st["w"], st["b"], st["threshold"]
n_want = args.n or mcfg["explain"]["n_samples"]
tol = mcfg["explain"]["tolerance"]

te = load_split(proc, args.dataset, "test")
X, y, wt, att = xyw(te, feats)
z = stack_logit(raw_matrix(models, X), w, b)
p = 1 / (1 + np.exp(-z))
flagged = np.flatnonzero(p >= thr)
log.info("%d/%d vector test bị gắn cờ (ngưỡng %.4f)", len(flagged), len(te), thr)

# lấy mẫu phân tầng theo tên tấn công trong số luồng bị gắn cờ
rng = np.random.default_rng(mcfg["seed"])
names = np.unique(att[flagged])
per = max(1, n_want // len(names))
pick = np.concatenate([rng.choice(flagged[att[flagged] == a], size=min(per, int((att[flagged] == a).sum())), replace=False)
                       for a in names])
pick = np.sort(pick)
Xs = X.iloc[pick]
log.info("Giải thích %d luồng (%d tên tấn công trong số bị gắn cờ)", len(pick), len(names))

ex = explain_stack(models, w, b, Xs)
err = {k: {"max": float(v.max()), "mean": float(v.mean())} for k, v in ex["additivity_error"].items()}
ok = all(v["max"] <= tol for v in err.values())
log.info("Kiểm tra tính cộng (tolerance %.0e): %s | %s", tol, "ĐẠT" if ok else "KHÔNG ĐẠT",
         {k: f"{v['max']:.2e}" for k, v in err.items()})

G_stack, gnames = group_attribution(ex["phi_stack"], feats, groups, GROUP_ORDER)
G_base = [group_attribution(ph, feats, groups, GROUP_ORDER)[0] for ph in ex["phi_per_base"]]
A = agreement_score(G_base)

out = pd.DataFrame(G_stack, columns=[f"phi_{g}" for g in gnames])
for k, n in enumerate(BASE_NAMES):
    for j, g in enumerate(gnames):
        out[f"phi_{n}_{g}"] = G_base[k][:, j]
out["z"], out["agreement"] = ex["z"], A
out["test_idx"] = pick  # chỉ số hàng trong tập test; bước 22 và 24 đọc lại để chạy trên đúng các luồng này
sub = te.iloc[pick].reset_index(drop=True)
for c in (LABEL_COL, ATTACK_COL, FAMILY_COL, WEIGHT_COL):
    out[c] = sub[c].to_numpy()
out.to_parquet(proc / args.dataset / "explain_sample.parquet", index=False)

fam = {}
for f_name, idx in out.groupby(FAMILY_COL, observed=True).groups.items():
    fam[str(f_name)] = {g: float(out.loc[idx, f"phi_{g}"].abs().mean()) for g in gnames}
summary = {"dataset": args.dataset, "n_explained": int(len(pick)), "n_flagged": int(len(flagged)),
           "threshold": thr, "meta_weights": dict(zip(BASE_NAMES, map(float, w))), "meta_bias": b,
           "additivity_error": err, "additivity_ok": ok, "groups": gnames,
           "mean_abs_group_attribution_by_family": fam,
           "agreement": {"mean": float(np.nanmean(A)), "p10": float(np.nanpercentile(A, 10)),
                         "p50": float(np.nanpercentile(A, 50)), "p90": float(np.nanpercentile(A, 90))}}
stem = f"explain_{args.dataset}" + (f"_{args.tag}" if args.tag else "")
write_json(summary, resolve_path(cfg, "reports_dir") / f"{stem}.json")
log.info("Đã ghi reports/%s.json và %s", stem, proc / args.dataset / "explain_sample.parquet")
print(explain_text(args.dataset, summary))
