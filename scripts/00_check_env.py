"""Kiểm tra môi trường Python và các thư viện cần cho từng giai đoạn."""
import importlib
import platform
import sys

import _bootstrap  # noqa: F401

STEP1 = ["numpy", "pandas", "sklearn", "scipy", "yaml", "pyarrow", "joblib", "pytest"]
LATER = ["xgboost", "lightgbm", "shap", "lime", "matplotlib"]


def check(names):
    ok = True
    for n in names:
        try:
            m = importlib.import_module(n)
            print(f"  [OK]    {n:<12} {getattr(m, '__version__', '')}")
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"  [THIẾU] {n:<12} ({type(e).__name__})")
    return ok


print(f"Python {sys.version.split()[0]} | {platform.system()} {platform.release()}")
print("Bước 1 (dữ liệu):")
a = check(STEP1)
print("Các bước sau (mô hình, giải thích):")
b = check(LATER)
print("\nKết quả:", "sẵn sàng cho bước 1" if a else "cần cài thêm: pip install -r requirements.txt")
if a and not b:
    print("Chưa cần các thư viện của bước sau, sẽ cài khi tới bước 3.")
