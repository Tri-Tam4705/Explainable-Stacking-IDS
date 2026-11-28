"""Hàm dùng chung cho các bước giải thích và kiểm tra lời giải thích.

- Lấy mẫu phân tầng theo tên tấn công, và mẫu con cố định của mẫu giải thích (bước 08) cho các phép kiểm tra tốn
  thời gian (scripts/22_explain_checks.py, 24_group_shapley.py).
- Hàm dự đoán của stacking trên mảng numpy: logit z và xác suất tấn công.
- Độ đo so sánh hai bộ đóng góp: tương quan hạng từng luồng, Jaccard top-k, độ quan trọng toàn cục.
- Phép xóa đặc trưng bằng giá trị nền (trung vị lành tính), dùng làm phép đối chiếu cho phép xóa trong phân phối.
Các hàm ở đây chỉ tính toán, không đọc ghi file.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import rankdata

from xsids.models.stacking import raw_matrix, stack_logit


# ── lấy mẫu luồng cần giải thích ────────────────────────────────────────────
def stratified_pick(candidates: np.ndarray, attack: np.ndarray, n_want: int, rng: np.random.Generator) -> np.ndarray:
    """Lấy mẫu phân tầng theo tên tấn công, cùng quy tắc với `scripts/08_explain.py`.

    Với cùng `candidates`, `n_want` và cùng trạng thái `rng` thì kết quả trùng đúng mẫu mà bước SHAP đã dùng,
    nên các phép kiểm tra chạy trên đúng những luồng của mẫu đã lưu.
    """
    candidates = np.asarray(candidates)
    names = np.unique(attack[candidates])
    per = max(1, n_want // len(names))
    parts = []
    for a in names:
        pool = candidates[attack[candidates] == a]
        parts.append(rng.choice(pool, size=min(per, len(pool)), replace=False))
    return np.sort(np.concatenate(parts))

def explain_subsample(explain_idx: np.ndarray, attack: np.ndarray, n_want: int, seed: int) -> np.ndarray:
    """Mẫu con cố định của mẫu giải thích (bước 08), phân tầng theo tên tấn công, cho các phép kiểm tra tốn thời gian.

    Cùng `explain_idx`, `n_want` và `seed` thì luôn ra cùng một tập luồng (đã sắp xếp), nên bước 22 và bước 24 chạy
    trên đúng những luồng giống nhau mà không cần lưu file trung gian.
    """
    pool = np.sort(np.asarray(explain_idx, dtype=int))
    return stratified_pick(pool, np.asarray(attack), min(int(n_want), len(pool)), np.random.default_rng(seed + 1))


def recover_sample_idx(z_all: np.ndarray, attack: np.ndarray, sample_z: np.ndarray, sample_attack: np.ndarray,
                       tol: float = 1e-6) -> np.ndarray | None:
    """Tìm lại chỉ số test của mẫu SHAP đã lưu khi file parquet cũ chưa có cột `test_idx`.

    Ghép từng dòng đã lưu với dòng test có cùng tên tấn công và logit z gần nhất (sai số < tol). Trả về None nếu có dòng
    không ghép được hoặc hai dòng ghép vào cùng một chỉ số, khi đó không được dùng kết quả (phải chạy lại bước 08).
    Cách này chịu được việc ngưỡng gắn cờ lệch 1 luồng giữa các lần chạy do sai số số thực của dự đoán đa luồng.
    """
    out = np.empty(len(sample_z), dtype=int)
    for a in np.unique(sample_attack):
        cand = np.flatnonzero(attack == a)
        order = np.argsort(z_all[cand])
        zs = z_all[cand][order]
        for i in np.flatnonzero(sample_attack == a):
            j = int(np.searchsorted(zs, sample_z[i]))
            near = [k for k in (j - 1, j) if 0 <= k < len(zs)]
            if not near:
                return None
            k = min(near, key=lambda q: abs(zs[q] - sample_z[i]))
            if abs(zs[k] - sample_z[i]) > tol:
                return None
            out[i] = cand[order[k]]
    return out if len(np.unique(out)) == len(out) else None


# ── hàm dự đoán của stacking ────────────────────────────────────────────────
def make_logit_fn(models: dict, w: np.ndarray, b: float, feature_names: list[str]):
    """A (n × d, numpy) -> z (n,), logit của stacking."""
    def fn(A) -> np.ndarray:
        X = pd.DataFrame(np.asarray(A, dtype="float32"), columns=feature_names)
        return stack_logit(raw_matrix(models, X), w, b)
    return fn


def make_proba_fn(models: dict, w: np.ndarray, b: float, feature_names: list[str]):
    """A (n × d, numpy) -> p (n,), xác suất tấn công."""
    logit = make_logit_fn(models, w, b, feature_names)
    return lambda A: expit(logit(A))


def binary_columns(X: pd.DataFrame) -> list[int]:
    """Chỉ số các cột chỉ nhận giá trị 0/1 (cờ TCP, is_tcp...). Phép nhiễu đầu vào giữ nguyên các cột này."""
    idx = []
    for i, c in enumerate(X.columns):
        u = np.unique(X[c].to_numpy())
        if len(u) <= 2 and set(u.tolist()) <= {0.0, 1.0}:
            idx.append(i)
    return idx


# ── so sánh hai bộ đóng góp ──────────────────────────────────────────────────
def summarize(v) -> dict:
    v = np.asarray(v, dtype="float64")
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return {"mean": None, "p10": None, "p50": None, "p90": None, "n": 0}
    return {"mean": float(v.mean()), "p10": float(np.percentile(v, 10)), "p50": float(np.percentile(v, 50)),
            "p90": float(np.percentile(v, 90)), "n": int(len(v))}


def rank_corr_rows(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Tương quan hạng Spearman theo từng hàng (hạng đồng điểm lấy trung bình). NaN nếu một hàng không đổi."""
    ra, rb = rankdata(A, axis=1), rankdata(B, axis=1)
    ra = ra - ra.mean(axis=1, keepdims=True)
    rb = rb - rb.mean(axis=1, keepdims=True)
    den = np.sqrt((ra * ra).sum(axis=1) * (rb * rb).sum(axis=1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.where(den > 0, (ra * rb).sum(axis=1) / np.where(den > 0, den, 1), np.nan)


def topk_sets(A: np.ndarray, k: int) -> np.ndarray:
    """Chỉ số k đặc trưng có |giá trị| lớn nhất mỗi hàng (n × k)."""
    return np.argsort(-np.abs(A), axis=1, kind="stable")[:, :k]


def topk_jaccard(A: np.ndarray, B: np.ndarray, k: int) -> np.ndarray:
    ta, tb = topk_sets(A, k), topk_sets(B, k)
    out = np.zeros(len(A))
    for i in range(len(A)):
        sa, sb = set(ta[i].tolist()), set(tb[i].tolist())
        out[i] = len(sa & sb) / len(sa | sb)
    return out


def global_importance(A: np.ndarray) -> np.ndarray:
    """Mean |giá trị| mỗi đặc trưng, chuẩn hóa tổng bằng 1 để so hai phương pháp khác thang đo."""
    m = np.abs(A).mean(axis=0)
    s = m.sum()
    return m / s if s > 0 else m


# ── phép xóa bằng giá trị nền ───────────────────────────────────────────────
def deletion_test(score_fn, X: np.ndarray, attributions: dict[str, np.ndarray], baseline: np.ndarray,
                  ks: list[int], threshold: float, rng: np.random.Generator, n_random: int = 5,
                  eligible_only: bool = True) -> dict:
    """Độ trung thực cục bộ bằng phép xóa: thay k đặc trưng đóng góp mạnh nhất về phía "tấn công" bằng giá trị nền.

    Giải thích càng đúng thì xác suất tấn công càng tụt mạnh. Trả về, với mỗi phương pháp và mỗi k:
        mean_drop : trung bình p_trước − p_sau
        flip_rate : tỷ lệ luồng (đều đang bị gắn cờ) rơi xuống dưới ngưỡng sau khi xóa
    `score_fn(A) -> p`. Dòng "random" là mốc đối chứng: xóa k đặc trưng ngẫu nhiên (trung bình n_random lần).

    eligible_only=True: chỉ xét các đặc trưng có giá trị KHÁC giá trị nền. Thay một đặc trưng đã bằng giá trị nền
    là phép không làm gì với mọi phương pháp, nên nếu không lọc thì phương pháp nào xếp hạng cao một đặc trưng như vậy
    sẽ bị phạt vì trùng hợp với giá trị nền (vd. cờ nhị phân luôn bằng 0 ở cả luồng tấn công lẫn trung vị lành tính).
    Bộ lọc áp dụng giống nhau cho mọi phương pháp, kể cả ngẫu nhiên.
    """
    n, d = X.shape
    p0 = score_fn(X)
    rows = np.arange(n)[:, None]
    eligible = (X != baseline) if eligible_only else np.ones((n, d), dtype=bool)

    def run(keys: np.ndarray, k: int) -> dict:
        cols = np.argsort(-np.where(eligible, keys, -np.inf), axis=1, kind="stable")[:, :k]
        Xk = X.copy()
        Xk[rows, cols] = baseline[cols]
        pk = score_fn(Xk)
        return {"mean_drop": float((p0 - pk).mean()), "flip_rate": float(((p0 >= threshold) & (pk < threshold)).mean())}

    out: dict = {str(k): {} for k in ks}
    for name, A in attributions.items():  # đóng góp dương lớn nhất trước
        for k in ks:
            out[str(k)][name] = run(A, k)
    for k in ks:
        runs = [run(rng.random((n, d)), k) for _ in range(n_random)]
        out[str(k)]["random"] = {m: float(np.mean([r[m] for r in runs])) for m in ("mean_drop", "flip_rate")}
    out["n_eligible_mean"] = float(eligible.sum(axis=1).mean())
    return out
