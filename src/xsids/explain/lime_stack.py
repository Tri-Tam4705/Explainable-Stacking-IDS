"""LIME cục bộ cho hệ stacking, đối chứng độc lập với TreeSHAP gộp ở `shap_stack.py`.

SHAP gộp khai thác cấu trúc cây và trọng số meta-learner (giải thích chính xác nhưng phụ thuộc kiến trúc).
LIME thì coi cả hệ stacking là hộp đen: chỉ gọi hàm dự đoán trên các điểm nhiễu quanh một luồng mạng rồi khớp
mô hình tuyến tính có trọng số theo khoảng cách. Nếu hai phương pháp độc lập cùng chỉ ra những đặc trưng giống
nhau thì giải thích đáng tin hơn; chỗ nào lệch nhau là chỗ cần xem lại.

Hai cách chọn đại lượng được giải thích (`target`):
    proba : LIME phân loại trên xác suất tấn công p = σ(z)        (mặc định, đúng nghĩa "gọi bộ dự đoán")
    logit : LIME hồi quy trên z = b + Σ wₖ·fₖ(x), cùng thang đo với SHAP (SHAP giải thích z)
Thứ hạng và dấu của đóng góp không đổi giữa hai cách vì σ đơn điệu, nhưng độ lớn thì khác thang đo,
nên các độ đo bên dưới chỉ so sánh thứ hạng, dấu và hiệu ứng khi xóa đặc trưng, không so giá trị tuyệt đối.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import rankdata

from xsids.models.stacking import raw_matrix, stack_logit

TARGETS = ("proba", "logit")


# ── lấy mẫu luồng cần giải thích ────────────────────────────────────────────
def stratified_pick(candidates: np.ndarray, attack: np.ndarray, n_want: int, rng: np.random.Generator) -> np.ndarray:
    """Lấy mẫu phân tầng theo tên tấn công, cùng quy tắc với `scripts/08_explain.py`.

    Với cùng `candidates`, `n_want` và cùng trạng thái `rng` thì kết quả trùng đúng mẫu mà bước SHAP đã dùng,
    nên LIME và SHAP được so sánh trên đúng những luồng giống nhau.
    """
    candidates = np.asarray(candidates)
    names = np.unique(attack[candidates])
    per = max(1, n_want // len(names))
    parts = []
    for a in names:
        pool = candidates[attack[candidates] == a]
        parts.append(rng.choice(pool, size=min(per, len(pool)), replace=False))
    return np.sort(np.concatenate(parts))


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


# ── hàm dự đoán của stacking cho LIME ───────────────────────────────────────
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


def make_lime_predict_fn(models: dict, w: np.ndarray, b: float, feature_names: list[str], target: str = "proba"):
    """Hàm dự đoán đúng định dạng LIME cần: (n, 2) xác suất [Benign, Attack] hoặc (n,) logit."""
    _check_target(target)
    if target == "logit":
        return make_logit_fn(models, w, b, feature_names)
    proba = make_proba_fn(models, w, b, feature_names)

    def fn(A) -> np.ndarray:
        p = proba(A)
        return np.column_stack([1.0 - p, p])
    return fn


def _check_target(target: str) -> None:
    if target not in TARGETS:
        raise ValueError(f"target phải là một trong {TARGETS}, nhận được {target!r}")


# ── explainer và chạy LIME ──────────────────────────────────────────────────
def binary_columns(X: pd.DataFrame) -> list[int]:
    """Chỉ số các cột chỉ nhận giá trị 0/1 (cờ TCP, is_tcp...). LIME xử lý chúng như đặc trưng phân loại."""
    idx = []
    for i, c in enumerate(X.columns):
        u = np.unique(X[c].to_numpy())
        if len(u) <= 2 and set(u.tolist()) <= {0.0, 1.0}:
            idx.append(i)
    return idx


DISCRETIZERS = ("quartile", "decile", "none")


def build_explainer(X_bg: pd.DataFrame, target: str = "proba", discretizer: str = "quartile",
                    kernel_width: float | None = None, seed: int = 0):
    """Dựng LimeTabularExplainer từ mẫu nền (lấy từ tập train, không dùng test để tránh rò rỉ).

    discretizer = "none": không rời rạc hóa đặc trưng liên tục. LIME khi đó lấy mẫu nhiễu Gaussian quanh luồng cần
    giải thích (sample_around_instance=True) và hệ số là độ dốc theo giá trị thật, không phải hệ số của chỉ báo
    "cùng khoảng (bin)". Cách này tránh việc cờ nhị phân luôn có |hệ số| lớn hơn chỉ báo bin của đặc trưng liên tục.
    """
    _check_target(target)
    if discretizer not in DISCRETIZERS:
        raise ValueError(f"discretizer phải là một trong {DISCRETIZERS}, nhận được {discretizer!r}")
    from lime.lime_tabular import LimeTabularExplainer

    disc = discretizer != "none"

    return LimeTabularExplainer(
        X_bg.to_numpy(dtype="float64"),
        feature_names=list(X_bg.columns),
        class_names=["Benign", "Attack"],
        categorical_features=binary_columns(X_bg),
        mode="classification" if target == "proba" else "regression",
        discretize_continuous=disc,
        discretizer=discretizer if disc else "quartile",
        kernel_width=kernel_width,
        feature_selection="none",      # giữ đủ d đặc trưng để so với SHAP; việc cắt top-k làm sau
        sample_around_instance=not disc,  # khi đã rời rạc hóa, LIME bỏ qua cờ này: luôn lấy mẫu theo phân phối train
        random_state=seed,
    )


def _pick(v, key):
    """`intercept` là dict theo nhãn; `score`/`local_pred` là số (bản lime mới hơn có thể là dict theo nhãn)."""
    return v[key] if isinstance(v, dict) else v


def lime_weights(explainer, predict_fn, X_rows: np.ndarray, target: str = "proba", num_samples: int = 5000,
                 progress=None) -> dict:
    """Chạy LIME cho từng hàng của X_rows (n × d). Trả về trọng số đầy đủ (n × d) và chỉ số khớp cục bộ.

    Trọng số dương nghĩa là đặc trưng (ở giá trị/khoảng giá trị hiện tại) đẩy dự đoán về phía tấn công.
        weights    : đóng góp của từng đặc trưng trong mô hình thay thế cục bộ: hệ số của chỉ báo (khi rời rạc hóa) hoặc
                     độ dốc × (x − mean)/scale (khi không rời rạc hóa), để cùng ý nghĩa "đóng góp" với SHAP
        intercept  : hệ số chặn của mô hình thay thế
        r2         : R² có trọng số của mô hình thay thế trên vùng lân cận (LIME `score`)
        local_pred : dự đoán của mô hình thay thế tại chính luồng đó
    """
    _check_target(target)
    # Lưu ý: ở chế độ hồi quy LIME để trọng số thật ở khóa 1 và bản ĐẢO DẤU ở khóa 0 (xem LimeTabularExplainer.explain_instance),
    # nên cả hai chế độ đều phải đọc khóa 1. Đọc nhầm khóa 0 sẽ đảo dấu toàn bộ giải thích.
    key = 1
    score_key = 1 if target == "proba" else 0
    X_rows = np.asarray(X_rows, dtype="float64")
    n, d = X_rows.shape
    W = np.zeros((n, d))
    intercept, r2, local_pred = np.zeros(n), np.zeros(n), np.zeros(n)
    # Khi không rời rạc hóa, LIME khớp mô hình tuyến tính trên dữ liệu đã chuẩn hóa (x − mean)/scale, nên hệ số là ĐỘ DỐC
    # theo đơn vị độ lệch chuẩn. SHAP là ĐÓNG GÓP (độ dốc × độ lệch so với giá trị nền), muốn so sánh/xóa đặc trưng
    # theo thứ hạng thì phải nhân hệ số với (x − mean)/scale. Đặc trưng categorical (chỉ báo = 1 tại luồng) đã là đóng góp.
    contrib_mult = None
    if getattr(explainer, "discretizer", None) is None:
        contrib_mult = (X_rows - explainer.scaler.mean_) / explainer.scaler.scale_
        for c in explainer.categorical_features:
            contrib_mult[:, c] = 1.0
    for i in range(n):
        exp = explainer.explain_instance(X_rows[i], predict_fn, labels=(1,), num_features=d, num_samples=num_samples)
        for j, wt in exp.local_exp[key]:
            W[i, j] = wt
        intercept[i] = float(np.ravel(_pick(exp.intercept, key))[0])
        r2[i] = float(_pick(exp.score, score_key))
        local_pred[i] = float(np.ravel(_pick(exp.local_pred, score_key))[0])
        if progress is not None:
            progress(i + 1, n)
    if contrib_mult is not None:
        W = W * contrib_mult
    return {"weights": W, "intercept": intercept, "r2": r2, "local_pred": local_pred}


# ── độ đo so sánh ───────────────────────────────────────────────────────────
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


def topk_sign_agreement(ref: np.ndarray, other: np.ndarray, k: int) -> np.ndarray:
    """Trong k đặc trưng quan trọng nhất của `ref`, tỷ lệ có cùng dấu đóng góp ở `other`."""
    top = topk_sets(ref, k)
    rows = np.arange(len(ref))[:, None]
    return (np.sign(ref[rows, top]) == np.sign(other[rows, top])).mean(axis=1)


def global_importance(A: np.ndarray) -> np.ndarray:
    """Mean |giá trị| mỗi đặc trưng, chuẩn hóa tổng bằng 1 để so hai phương pháp khác thang đo."""
    m = np.abs(A).mean(axis=0)
    s = m.sum()
    return m / s if s > 0 else m


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


def stability_metrics(runs: np.ndarray, k: int) -> dict:
    """Độ ổn định của LIME khi lặp lại với seed khác nhau. `runs` có dạng (R, n, d).

    Với mỗi luồng, so sánh mọi cặp lần chạy: Spearman của |trọng số| và Jaccard của top-k.
    TreeSHAP xác định (không ngẫu nhiên) nên đạt 1.0 theo cấu trúc; đây là điểm LIME bị thiệt.
    """
    R, n, _ = runs.shape
    rho, jac, top1 = [], [], []
    for a in range(R):
        for c in range(a + 1, R):
            rho.append(rank_corr_rows(np.abs(runs[a]), np.abs(runs[c])))
            jac.append(topk_jaccard(runs[a], runs[c], k))
            top1.append((np.argmax(np.abs(runs[a]), axis=1) == np.argmax(np.abs(runs[c]), axis=1)).astype(float))
    return {"repeats": int(R), "n_flows": int(n),
            "spearman": summarize(np.nanmean(np.stack(rho), axis=0)),
            "topk_jaccard": summarize(np.mean(np.stack(jac), axis=0)),
            "top1_same_rate": float(np.mean(np.stack(top1)))}
