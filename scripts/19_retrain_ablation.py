"""Ablation CÓ huấn luyện lại: bỏ một nhóm (hoặc một tập) đặc trưng, huấn luyện lại cả hệ, đánh giá trong miền và chéo bộ.

Khác bước 7 (10_ablation.py) ở chỗ mô hình được huấn luyện lại không có các đặc trưng đó, nên kết quả đo GIÁ TRỊ của đặc trưng
(mô hình có bù được bằng đặc trưng khác không), không phải độ nhạy của mô hình đã huấn luyện.

Các cấu hình:
    full                   đủ 25 đặc trưng (phải trùng kết quả chính; dùng làm mốc)
    drop_<nhóm>            bỏ một trong sáu nhóm ngữ nghĩa
    drop_server_psh        bỏ cờ server_psh (trên UNSW: bằng 1 ở 24,8% lành tính, 0% tấn công)
    drop_suspects          bỏ server_psh, src_to_dst_bytes_per_s, src_to_dst_throughput: ba đặc trưng mà mô hình UNSW
                           phụ thuộc nặng nhất và nghi là đặc điểm của testbed
Mỗi cấu hình được đánh giá trên test của chính bộ đó và trên test của bộ còn lại (ngưỡng của bộ nguồn).

Dùng:
    python scripts/19_retrain_ablation.py --dataset unsw
    python scripts/19_retrain_ablation.py --dataset cse --fast
Đầu ra: reports/retrain_ablation_<ds>.json. Không ghi đè mô hình hay kết quả chính.
Thời gian: mỗi cấu hình bằng bước 3 + 4 (khoảng 3 phút với UNSW, 2 giờ với CSE), nhân 9 cấu hình.
"""
import argparse
import time

import _bootstrap  # noqa: F401
from xsids.config import load_config, resolve_path
from xsids.features.engineer import GROUP_ORDER
from xsids.models.data import load_feature_set, load_split, natural_prevalence, xyw
from xsids.models.experiment import evaluate_system, fit_system, thresholds
from xsids.models.zoo import get_params, load_models_cfg
from xsids.utils import get_logger, write_json

SUSPECTS = ["server_psh", "src_to_dst_bytes_per_s", "src_to_dst_throughput"]

log = get_logger()
ap = argparse.ArgumentParser()
ap.add_argument("--dataset", choices=["unsw", "cse"], required=True)
ap.add_argument("--fast", action="store_true")
ap.add_argument("--only", nargs="+", default=None, help="chỉ chạy các cấu hình này (vd. full drop_suspects)")
args = ap.parse_args()

cfg, mcfg = load_config(), load_models_cfg()
ds, other = args.dataset, ("cse" if args.dataset == "unsw" else "unsw")
proc, rdir = resolve_path(cfg, "processed_dir"), resolve_path(cfg, "reports_dir")
fs = load_feature_set(proc)
feats, groups = fs["features"], fs["groups"]
seed, n_jobs, folds, C = mcfg["seed"], mcfg["n_jobs"], mcfg["cv_folds"], mcfg["meta"]["C"]
params = get_params(mcfg, args.fast)

tr = load_split(proc, ds, "train")
va, te, te_o = load_split(proc, ds, "val"), load_split(proc, ds, "test"), load_split(proc, other, "test")
pi, pi_o = natural_prevalence(proc, ds), natural_prevalence(proc, other)

configs = {"full": []}
configs.update({f"drop_{g}": [f for f in feats if groups[f] == g] for g in GROUP_ORDER})
configs["drop_server_psh"] = ["server_psh"]
configs["drop_suspects"] = SUSPECTS
if args.only:
    configs = {k: v for k, v in configs.items() if k in args.only}

out = {"dataset": ds, "cross_dataset": other, "fast": args.fast, "n_features": len(feats), "configs": {}}
for name, dropped in configs.items():
    t0 = time.time()
    keep = [f for f in feats if f not in dropped]
    Xtr, ytr, _, atr = xyw(tr, keep)
    Xva, yva, wva, _ = xyw(va, keep)
    Xte, yte, wte, ate = xyw(te, keep)
    Xo, yo, wo, ao = xyw(te_o, keep)
    system = fit_system(Xtr, ytr, atr, params, seed, n_jobs, folds, C)
    thr = thresholds(system, Xva, yva, wva)
    r_in = evaluate_system(system, thr, Xte, yte, wte, ate, pi, per_attack=True)
    r_x = evaluate_system(system, thr, Xo, yo, wo, ao, pi_o)
    out["configs"][name] = {"dropped": dropped, "n_kept": len(keep), "in_domain": r_in, "cross": r_x,
                            "meta_weights": {n: float(v) for n, v in zip(("rf", "et", "xgb", "lgbm"), system["w"])},
                            "seconds": round(time.time() - t0, 1)}
    s, x = r_in["stack_explainable"], r_x["stack_explainable"]
    log.info("%-22s (%2d đặc trưng, %.0fs) trong miền DR %.4f FAR %.5f F1 %.4f | chéo DR %.4f FAR %.5f ROC-AUC %.3f",
             name, len(keep), time.time() - t0, s["dr"], s["far"], s["macro_f1"], x["dr"], x["far"], x["roc_auc"])
    write_json(out, rdir / f"retrain_ablation_{ds}.json")  # ghi sau mỗi cấu hình để không mất kết quả nếu dừng giữa chừng

base = out["configs"].get("full", {}).get("in_domain", {}).get("stack_explainable")
print(f"\nABLATION CÓ HUẤN LUYỆN LẠI — {ds.upper()} (stacking; chéo = kiểm tra trên {other.upper()} với ngưỡng nguồn)")
print("─" * 118)
print(f"  {'Cấu hình':20s}{'Số đ.trưng':>11s}{'DR':>9s}{'FAR':>10s}{'macro-F1':>10s}{'PR-AUC':>9s}{'ΔmacroF1':>10s}"
      f"{'Chéo DR':>10s}{'Chéo FAR':>10s}{'Chéo ROC-AUC':>14s}")
for name, c in out["configs"].items():
    s, x = c["in_domain"]["stack_explainable"], c["cross"]["stack_explainable"]
    d = f"{s['macro_f1'] - base['macro_f1']:+.4f}" if base and name != "full" else "-"
    print(f"  {name:20s}{c['n_kept']:>11d}{s['dr']:>9.4f}{s['far']:>10.5f}{s['macro_f1']:>10.4f}{s['pr_auc']:>9.4f}{d:>10s}"
          f"{x['dr']:>10.4f}{x['far']:>10.5f}{x['roc_auc']:>14.3f}")
print("  Mô hình được huấn luyện lại không có các đặc trưng bị bỏ; ngưỡng chọn lại trên val.")
log.info("Đã ghi %s", rdir / f"retrain_ablation_{ds}.json")
