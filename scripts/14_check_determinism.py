"""Chẩn đoán: mô hình thành phần nào cho kết quả khác nhau giữa các lần chạy dù cùng dữ liệu, seed và tham số?

Với mỗi mô hình, chạy lại bước out-of-fold (OOF) của bước 4 hai lần rồi so từng dự đoán. Nếu có `--vs-single`, chạy thêm
một lần với 1 luồng để biết chênh lệch có do đa luồng hay không. Không ghi đè mô hình hay báo cáo nào.

Ví dụ:
    python scripts/14_check_determinism.py --dataset unsw
    python scripts/14_check_determinism.py --dataset unsw --models xgb lgbm --vs-single
"""
import argparse
import time
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
from xsids.config import load_config, resolve_path
from xsids.models.data import cap_train, load_feature_set, load_split, xyw
from xsids.models.determinism import meta_weight_spread, oof_repeat_diff
from xsids.models.zoo import BASE_NAMES, get_params, load_models_cfg
from xsids.report.console import heading, num, table
from xsids.utils import get_logger

log = get_logger()
ap = argparse.ArgumentParser()
ap.add_argument("--dataset", choices=["unsw", "cse"], required=True)
ap.add_argument("--models", nargs="+", choices=BASE_NAMES, default=list(BASE_NAMES))
ap.add_argument("--repeats", type=int, default=2, help="số lần chạy lặp lại với cùng cấu hình (tối thiểu 2)")
ap.add_argument("--vs-single", action="store_true", help="chạy thêm một lần với 1 luồng để so sánh")
ap.add_argument("--rows", type=int, default=None, help="chỉ dùng ngẫu nhiên N vector train để chạy nhanh")
ap.add_argument("--processed-dir", default=None)
args = ap.parse_args()

cfg, mcfg = load_config(), load_models_cfg()
proc = Path(args.processed_dir) if args.processed_dir else resolve_path(cfg, "processed_dir")
feats = load_feature_set(proc)["features"]
seed, n_jobs, folds = mcfg["seed"], mcfg["n_jobs"], mcfg["cv_folds"]
params = get_params(mcfg, False)

tr = cap_train(load_split(proc, args.dataset, "train"), mcfg["train_cap"].get(args.dataset), seed)
if args.rows and args.rows < len(tr):
    tr = tr.sample(n=args.rows, random_state=seed).reset_index(drop=True)
X, y, _, att = xyw(tr, feats)
log.info("%s: %d vector train, n_jobs=%s, %d fold, %d lần lặp", args.dataset, len(X), n_jobs, folds, args.repeats)

print(heading("THAM SỐ ĐANG ĐƯỢC DÙNG (đọc từ configs/models.yaml)"))
for n in args.models:
    print(f"  {n:5s} {params[n]}")

ROUNDING = 1e-9   # dưới ngưỡng này chỉ là sai số làm tròn số thực (thứ tự cộng khác nhau giữa các luồng)


def level(d: float) -> str:
    return "tất định" if d == 0 else ("lệch cỡ làm tròn" if d < ROUNDING else "KHÔNG tất định")


rows, F_runs, worst = [], {}, {}
for n in args.models:
    t = time.time()
    r = oof_repeat_diff(n, params[n], X, y, att, folds, seed, n_jobs, args.repeats, 1 if args.vs_single else None)
    F_runs[n] = r["runs"]
    rep, ref = r["repeat"], r["reference"]
    row = [n, level(rep["max_abs_diff"]), f"{rep['max_abs_diff']:.2e}", f"{rep['n_rows_differ']}/{rep['n_rows']}"]
    if args.vs_single:
        row += [f"{ref['max_abs_diff']:.2e}", level(ref["max_abs_diff"])]
    worst[n] = rep["max_abs_diff"]
    rows.append(row)
    log.info("  %s xong (%.0fs)", n, time.time() - t)

headers = ["Mô hình", "Giữa các lần lặp", "Lệch tối đa", "Số dòng lệch"] + (["Lệch so với 1 luồng", "Mức"] if args.vs_single else [])
print(heading("KẾT QUẢ: so sánh OOF giữa các lần chạy"))
print(table(headers, rows, "<<>>" + ("><" if args.vs_single else "")))

bad = [n for n, d in worst.items() if d >= ROUNDING]
rounding = [n for n, d in worst.items() if 0 < d < ROUNDING]
spread = None
if len(args.models) == len(BASE_NAMES):
    reps = min(len(F_runs[n]) for n in BASE_NAMES)
    F_list = [np.column_stack([F_runs[n][i] for n in BASE_NAMES]) for i in range(reps)]
    sp = meta_weight_spread(F_list, y, mcfg["meta"]["C"])
    print(heading("ẢNH HƯỞNG LÊN META-LEARNER (học lại trên từng lần lặp)"))
    print(table(["Lần"] + list(BASE_NAMES), [[i + 1] + [num(v, 4, sign=True) for v in w] for i, w in enumerate(sp["weights"])], ">" + ">" * len(BASE_NAMES)))
    spread = sp["max_abs_spread"]
    print(f"  Chênh lệch tối đa giữa các lần của một trọng số: {spread:.4f}")

print(heading("KẾT LUẬN"))
if bad:
    print(f"  Không tất định đáng kể: {', '.join(bad)} (lệch ≥ {ROUNDING:.0e}).")
    print("  Thử n_jobs: 1 trong configs/models.yaml để loại trừ chạy đa luồng (chạy chậm hơn); với LightGBM có thể thêm")
    print("  deterministic: true và force_row_wise: true.")
elif rounding:
    print(f"  {', '.join(rounding)} chỉ lệch cỡ làm tròn số thực (< {ROUNDING:.0e}); các mô hình còn lại giống hệt giữa các lần.")
    if spread is not None and spread < 1e-3:
        print("  Mức lệch này không làm đổi trọng số meta-learner trong thử nghiệm, nên KHÔNG giải thích được các dao động lớn hơn")
        print("  đã gặp khi chạy lại bước 07. Thử tăng --repeats (ví dụ 6) để có nhiều cơ hội quan sát hơn.")
else:
    print("  Mọi mô hình đã chọn đều tất định trong thử nghiệm này (cùng dữ liệu, seed, tham số và số luồng).")
if len(args.models) == len(BASE_NAMES) and spread is not None and spread >= 1e-3:
    print(f"  Trọng số meta-learner đổi tới {spread:.4f} giữa các lần lặp: dao động này tái hiện được trong thử nghiệm.")
