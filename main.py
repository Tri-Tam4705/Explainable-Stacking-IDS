"""Điểm vào tiện lợi cho toàn bộ hệ thống.

    python main.py check                         # kiểm tra môi trường
    python main.py step1 [--skip-peek]           # peek + audit + làm sạch + chia tập (cả hai dataset)
    python main.py step2                         # đặc trưng chung cho cả hai dataset
    python main.py models --fast                 # chạy thử nhanh: mô hình đơn -> stacking -> gộp SHAP
    python main.py models                        # chạy đầy đủ (lâu hơn nhiều)
    python main.py models --datasets unsw        # chỉ một dataset
    python main.py eval                          # bước 6+7: stability + ablation (cả hai dataset)
    python main.py eval --datasets unsw          # chỉ một dataset
    python main.py cross                         # bước 8: cross-dataset + LOFO
    python main.py shift                         # bước 8b: dịch chuyển phân phối giữa hai bộ, ngưỡng chỉnh lại
    python main.py report                        # bảng tổng hợp so sánh + biểu đồ từ reports/ (chạy được bất kỳ lúc nào, vài giây)
    python main.py all                           # check -> step1 -> step2 -> models -> eval -> cross -> shift -> report
    python main.py all --from eval               # chạy tiếp từ bước eval (bỏ qua các bước đã có kết quả)
    python main.py all --fast --datasets unsw    # chạy thử cả hệ thống với tham số nhỏ

Các lệnh tùy chọn, KHÔNG nằm trong `all` (không ghi đè mô hình và kết quả chính):
    python main.py multiseed                     # huấn luyện lại bước 3+4 với nhiều seed (cả hai dataset)
    python main.py diagnose                      # chẩn đoán: trọng số meta-learner âm, tấn công bị bỏ sót
    python main.py retrain                       # ablation có huấn luyện lại: bỏ từng nhóm đặc trưng, đánh giá trong miền và chéo bộ
    python main.py multisplit                    # nhiều phép chia train/val/test, kiểm định ghép cặp stacking so với từng mô hình đơn
    python main.py trust                         # A(x) có báo trước cảnh báo sai không; SHAP gộp của tấn công bị bỏ sót
    python main.py explaincheck                  # kiểm tra bổ sung cho lời giải thích: hàm giá trị, xóa trong phân phối, nhiễu, seed, nhãn xáo trộn
    python main.py groupshap                     # Shapley nhóm trong phân phối: độ trung thực so với oracle
    python main.py attacktype                    # bước hai đa lớp: gọi tên loại tấn công cho luồng bị gắn cờ
    python main.py labelaudit                    # kiểm toán nhãn: trùng vector khác nhãn, học tự tin, ảnh hưởng tới kết quả
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
S = ROOT / "scripts"
BOTH = ["unsw", "cse"]


def run(script: str, *args: str) -> None:
    cmd = [sys.executable, str(S / script), *args]
    print("\n>>>", " ".join(cmd[1:]), flush=True)
    subprocess.run(cmd, check=True)


PIPELINE = ["check", "step1", "step2", "models", "eval", "cross", "shift", "report"]
OPTIONAL = ["multiseed", "diagnose", "retrain", "multisplit", "trust", "explaincheck", "groupshap",
            "attacktype", "labelaudit"]

ap = argparse.ArgumentParser()
ap.add_argument("command", choices=PIPELINE + ["all"] + OPTIONAL)
ap.add_argument("--from", dest="start", choices=PIPELINE, default="check",
                help="với lệnh all: bắt đầu từ bước này (mặc định: check)")
ap.add_argument("--datasets", nargs="+", choices=BOTH, default=BOTH)
ap.add_argument("--sample-frac", type=float, default=None)
ap.add_argument("--skip-peek", action="store_true")
ap.add_argument("--fast", action="store_true", help="tham số mô hình nhỏ để chạy thử")
ap.add_argument("--n-explain", type=int, default=None, help="số luồng được giải thích (bước SHAP)")
ap.add_argument("--n-bootstrap", type=int, default=200, help="số lần bootstrap tập luồng cho stability (bước 6)")
ap.add_argument("--fill", choices=["median", "zero"], default=None, help="giá trị thay cho đặc trưng bị cắt (ablation, LOFO)")
ap.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 8)), help="các seed cho lệnh multiseed")
ap.add_argument("--splits", type=int, nargs="+", default=list(range(1, 11)), help="seed của các phép chia cho lệnh multisplit")
args = ap.parse_args()

fill = ["--fill", args.fill] if args.fill else []
need_both = set(args.datasets) != set(BOTH)

if args.command == "all":
    flags = ["--datasets", *args.datasets, "--n-bootstrap", str(args.n_bootstrap)]
    flags += ["--fast"] if args.fast else []
    flags += ["--skip-peek"] if args.skip_peek else []
    for opt, val in (("--sample-frac", args.sample_frac), ("--n-explain", args.n_explain), ("--fill", args.fill)):
        flags += [opt, str(val)] if val is not None else []
    steps = PIPELINE[PIPELINE.index(args.start):]
    print("Chạy cả hệ thống:", " -> ".join(steps))
    for step in steps:
        r = subprocess.run([sys.executable, str(Path(__file__).resolve()), step, *flags])
        if r.returncode != 0:
            sys.exit(f"\nDừng ở bước '{step}' (mã lỗi {r.returncode}). Sửa lỗi rồi chạy tiếp bằng: python main.py all --from {step}")
    print("\nĐã chạy xong cả hệ thống. Kết quả trong reports/, data/processed/ và models/.")
elif args.command == "check":
    run("00_check_env.py")
elif args.command == "step1":
    extra = ["--sample-frac", str(args.sample_frac)] if args.sample_frac is not None else []
    for ds in args.datasets:
        if not args.skip_peek:
            run("01b_peek_csv.py", "--dataset", ds)
        run("02_audit_data.py", "--dataset", ds, *extra)
        run("03_prepare_data.py", "--dataset", ds, *extra)
    print("\nXong bước 1. Xem reports/ và data/processed/.")
elif args.command == "step2":
    run("05_build_features.py", "--datasets", *args.datasets)
    print("\nXong bước 2. Xem reports/features.md.")
elif args.command == "eval":
    nb = ["--n-bootstrap", str(args.n_bootstrap)]
    for ds in args.datasets:
        run("09_stability.py", "--dataset", ds, *nb)
        run("10_ablation.py", "--dataset", ds, *fill)
    print("\nXong bước 6+7. Xem reports/stability_*.json và ablation_*.json.")
elif args.command in ("cross", "shift"):
    if need_both:  # hai bước này so hai bộ dữ liệu với nhau nên cần mô hình và dữ liệu của cả hai
        print(f"\nBỏ qua bước {args.command}: cần cả hai dataset (unsw và cse), đang chạy với --datasets {' '.join(args.datasets)}.")
    elif args.command == "cross":
        run("11_cross_lofo.py", *fill)
        print("\nXong bước 8. Xem reports/cross_lofo.json.")
    else:
        run("16_shift.py")
        print("\nXong bước 8b. Xem reports/shift.json.")
elif args.command == "report":
    run("13_summary.py", "--datasets", *args.datasets)
    print("\nXong bước 9. Xem reports/summary.txt và reports/figures/fig_*.png.")
elif args.command == "multiseed":
    fast = ["--fast"] if args.fast else []
    for ds in args.datasets:
        run("17_multiseed.py", "--dataset", ds, "--seeds", *map(str, args.seeds), *fast)
    print("\nXong. Xem reports/multiseed_*.json; chạy python main.py report để đưa vào bảng tổng hợp.")
elif args.command == "diagnose":
    fast = ["--fast"] if args.fast else []
    for ds in args.datasets:
        run("18_diagnostics.py", "--dataset", ds, *fast)
    print("\nXong. Xem reports/diagnostics_*.json.")
elif args.command == "retrain":
    fast = ["--fast"] if args.fast else []
    for ds in args.datasets:
        run("19_retrain_ablation.py", "--dataset", ds, *fast)
    print("\nXong. Xem reports/retrain_ablation_*.json.")
elif args.command == "multisplit":
    fast = ["--fast"] if args.fast else []
    for ds in args.datasets:
        run("20_multisplit.py", "--dataset", ds, "--splits", *map(str, args.splits), *fast)
    print("\nXong. Xem reports/multisplit_*.json.")
elif args.command == "trust":
    for ds in args.datasets:
        run("21_trust_missed.py", "--dataset", ds)
    print("\nXong. Xem reports/trust_*.json; chạy python main.py report để vẽ fig_trust.png và fig_missed.png.")
elif args.command == "explaincheck":
    fast = ["--fast"] if args.fast else []
    for ds in args.datasets:
        run("22_explain_checks.py", "--dataset", ds, *fast)
        run("23_sanity_randomization.py", "--dataset", ds, *fast)
    print("\nXong. Xem reports/explain_checks_*.json, sanity_*.json; chạy python main.py report để vẽ hình.")
elif args.command == "groupshap":
    fast = ["--fast"] if args.fast else []
    for ds in args.datasets:
        run("24_group_shapley.py", "--dataset", ds, *fast)
    print("\nXong. Xem reports/group_shapley_*.json; chạy python main.py report để đưa vào bảng tổng hợp.")
elif args.command == "attacktype":
    fast = ["--fast"] if args.fast else []
    for ds in args.datasets:
        run("26_attack_type.py", "--dataset", ds, *fast)
    print("\nXong. Xem reports/attack_type_*.json.")
elif args.command == "labelaudit":
    fast = ["--fast"] if args.fast else []
    for ds in args.datasets:
        run("27_label_audit.py", "--dataset", ds, "--retrain", *fast)
    print("\nXong. Xem reports/label_audit_*.json.")
else:  # models
    fast = ["--fast"] if args.fast else []
    n = ["--n", str(args.n_explain)] if args.n_explain else []
    for ds in args.datasets:
        run("06_baselines.py", "--dataset", ds, *fast)
        run("07_stacking.py", "--dataset", ds, *fast)
        run("08_explain.py", "--dataset", ds, *n)
    print("\nXong. Xem reports/baselines_*.json, stacking_*.md, explain_*.json.")
