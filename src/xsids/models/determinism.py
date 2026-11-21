"""Đo độ không tất định của bước huấn luyện: chạy lại OOF của một mô hình với đúng cùng dữ liệu, seed, tham số
rồi so sánh từng dự đoán. Dùng để biết mô hình nào cho kết quả khác nhau giữa các lần chạy (thường do đa luồng).
"""
from __future__ import annotations

import numpy as np

from xsids.models.stacking import fit_meta, oof_raw


def oof_repeat_diff(name: str, params: dict, X, y, attack, folds: int, seed: int, n_jobs: int, repeats: int = 2,
                    ref_n_jobs: int | None = None) -> dict:
    """Chạy oof_raw `repeats` lần (cùng n_jobs) và so các lần với lần đầu.

    ref_n_jobs: nếu có, chạy thêm một lần với số luồng này (thường 1) và so với lần đầu.
    Trả về dict: runs (list mảng OOF), repeat (thống kê chênh lệch giữa các lần lặp), reference (hoặc None).
    """
    runs = [oof_raw(name, params, X, y, attack, folds, seed, n_jobs) for _ in range(max(2, repeats))]

    def stats(a: np.ndarray, b: np.ndarray) -> dict:
        d = np.abs(a - b)
        return {"max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean()), "n_rows_differ": int((d > 0).sum()),
                "n_rows": int(len(d)), "identical": bool(d.max() == 0.0)}

    worst = max((stats(runs[0], r) for r in runs[1:]), key=lambda s: s["max_abs_diff"])
    ref = None
    if ref_n_jobs is not None:
        single = oof_raw(name, params, X, y, attack, folds, seed, ref_n_jobs)
        ref = stats(runs[0], single)
    return {"runs": runs, "repeat": worst, "reference": ref}


def meta_weight_spread(F_list: list[np.ndarray], y: np.ndarray, C: float) -> dict:
    """Học meta-learner trên từng ma trận OOF (mỗi lần lặp một ma trận) và đo độ lệch trọng số giữa các lần."""
    ws = np.array([fit_meta(F, y, C)[0] for F in F_list])
    return {"weights": ws, "max_abs_spread": float(np.abs(ws - ws[0]).max())}
