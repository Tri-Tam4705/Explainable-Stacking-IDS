"""Kiểm tra tỉnh táo cho lời giải thích bằng ngẫu nhiên hóa nhãn (data randomization test, Adebayo và cộng sự, 2018).

Xáo trộn nhãn của tập train, huấn luyện lại toàn bộ hệ (4 mô hình + meta-learner, cùng tham số và seed), rồi tính
SHAP gộp trên cùng tập luồng mà bước 22 dùng cho phép so qua seed. Một phương pháp giải thích đúng nghĩa phải cho
lời giải thích khác hẳn khi mô hình không học được gì; nếu lời giải thích gần như không đổi thì nó phản ánh cấu trúc
dữ liệu chứ không phản ánh mô hình. Mốc so sánh: độ giống nhau giữa các seed (bước 22, phần seeds).

Dùng:
    python scripts/23_sanity_randomization.py --dataset unsw
    python scripts/23_sanity_randomization.py --dataset cse --fast
Đầu ra: reports/sanity_<ds>.json. Không lưu mô hình, không ghi đè kết quả chính.
"""
import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import _bootstrap  # noqa: F401
from xsids.config import load_config, resolve_path
from xsids.explain.checks import compare, global_importance, pick_common
from xsids.explain.shap_stack import explain_stack, group_attribution
from xsids.features.engineer import GROUP_ORDER
from xsids.models.data import cap_train, load_feature_set, load_split, xyw
from xsids.models.experiment import fit_system, scores
from xsids.models.zoo import BASE_NAMES, get_params, load_models_cfg
from xsids.utils import get_logger, write_json

log = get_logger()
ap = argparse.ArgumentParser()
ap.add_argument("--dataset", choices=["unsw", "cse"], required=True)
ap.add_argument("--n", type=int, default=300, help="số luồng (giống --n-seed-flows của bước 22)")
ap.add_argument("--fast", action="store_true")
args = ap.parse_args()

cfg, mcfg = load_config(), load_models_cfg()
ds, seed, K = args.dataset, mcfg["seed"], mcfg["lime"]["top_k"]
root, proc, rdir = Path(cfg["_root"]), resolve_path(cfg, "processed_dir"), resolve_path(cfg, "reports_dir")
fs = load_feature_set(proc)
feats, groups = fs["features"], fs["groups"]
n = 20 if args.fast else args.n

te = load_split(proc, ds, "test")
X, y, wt, att = xyw(te, feats)
es = pd.read_parquet(proc / ds / "explain_sample.parquet")
common = pick_common(es["test_idx"].to_numpy(dtype=int), att, n, seed)
Xc = X.iloc[common]

mdir = root / "models" / ds
st = joblib.load(mdir / "stack.joblib")
real = explain_stack({m: joblib.load(mdir / f"{m}.joblib") for m in BASE_NAMES}, np.asarray(st["w"]), float(st["b"]), Xc)

t0 = time.time()
tr = cap_train(load_split(proc, ds, "train"), mcfg["train_cap"].get(ds), seed)
Xtr, ytr, _, _ = xyw(tr, feats)
rng = np.random.default_rng(seed + 40)
y_perm = rng.permutation(ytr)
strata = np.where(y_perm == 1, "attack", "benign")  # nhãn giả cũng dùng để phân tầng fold OOF
system = fit_system(Xtr, y_perm, strata, get_params(mcfg, args.fast), seed, mcfg["n_jobs"], mcfg["cv_folds"], mcfg["meta"]["C"])
s = scores(system, X)
auc = float(roc_auc_score(y, s["stack_explainable"], sample_weight=wt))
auc_u = float(roc_auc_score(y, s["stack_explainable"]))  # không trọng số: vài vector có n_orig rất lớn chi phối bản có trọng số
log.info("Hệ huấn luyện trên nhãn xáo trộn: ROC-AUC trên test %.3f có trọng số, %.3f không trọng số (%.0fs)", auc, auc_u, time.time() - t0)
rand = explain_stack(system["models"], np.asarray(system["w"]), float(system["b"]), Xc)


def grp(phi):
    return group_attribution(phi, feats, groups, GROUP_ORDER)[0]


out = {"dataset": ds, "fast": args.fast, "n_flows": int(len(common)), "roc_auc_random_labels": auc, "roc_auc_random_labels_unweighted": auc_u,
       "meta_weights_random": dict(zip(BASE_NAMES, map(float, system["w"]))),
       "feature": compare(real["phi_stack"], rand["phi_stack"], K), "group": compare(grp(real["phi_stack"]), grp(rand["phi_stack"]), 1),
       "global_group_importance": {"real": global_importance(grp(real["phi_stack"])).tolist(),
                                   "random": global_importance(grp(rand["phi_stack"])).tolist()},
       "mean_abs_phi_sum": {"real": float(np.abs(real["phi_stack"]).sum(1).mean()), "random": float(np.abs(rand["phi_stack"]).sum(1).mean())},
       "seconds": round(time.time() - t0, 1)}
write_json(out, rdir / f"sanity_{ds}.json")
log.info("Mô hình thật so với nhãn xáo trộn: Spearman từng luồng %.3f, top-5 Jaccard %.3f, τ nhóm toàn cục %.3f",
         out["feature"]["spearman_flow_mean"], out["feature"]["topk_jaccard_mean"], out["group"]["global_kendall"])
log.info("Đã ghi reports/sanity_%s.json", ds)
