"""Gộp giải thích TreeSHAP của các mô hình thành phần theo trọng số của meta-learner.

Vì z(x) = b + Σ wₖ·fₖ(x) tuyến tính và giá trị Shapley tuyến tính theo hàm được giải thích:
    φᵢ(z) = Σ wₖ·φᵢ(fₖ),   φ₀(z) = b + Σ wₖ·φ₀(fₖ)
nên giải thích của cả hệ stacking là CHÍNH XÁC (theo hàm giá trị của TreeSHAP từng mô hình).

Hai hàm giá trị:
  - mặc định (background=None): TreeSHAP "path-dependent". Mỗi mô hình coi một đặc trưng là vắng mặt theo thống kê
    huấn luyện của chính nó, nên SHAP gộp là giá trị Shapley của một trò chơi pha trộn bốn định nghĩa.
  - background=X_nền: TreeSHAP "interventional" với CÙNG một tập nền cho cả bốn mô hình. Khi đó
    v_z(S) = b + Σ wₖ·E_nền[fₖ(x_S, X_nền,S̄)] chính là hàm giá trị interventional của z, nên SHAP gộp bằng đúng
    SHAP interventional của cả hệ, với φ₀ = trung bình z trên tập nền.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import rankdata

from xsids.models.zoo import BASE_NAMES, raw_output


def _positive_class(sv) -> np.ndarray:
    if isinstance(sv, list):
        return np.asarray(sv[-1], dtype="float64")
    sv = np.asarray(sv, dtype="float64")
    return sv[:, :, -1] if sv.ndim == 3 else sv


def _base_value(ev) -> float:
    return float(np.atleast_1d(np.asarray(ev, dtype="float64")).ravel()[-1])


INTERVENTIONAL_BG_MAX = 100  # số luồng nền tối đa của SHAP interventional (lấy mẫu tất định từ tập nền truyền vào)


def tree_shap(model, X, background=None):
    """(φ theo đặc trưng cho lớp tấn công, giá trị nền) của một mô hình dạng cây.

    background=None: path-dependent. background = DataFrame: interventional trên đầu ra thô của mô hình.
    """
    import shap

    if background is None:
        expl = shap.TreeExplainer(model)
    else:
        # Tập nền thực dùng: tối đa INTERVENTIONAL_BG_MAX luồng. Đây là mặc định của shap (maskers.Independent lấy mẫu
        # tất định, random_state = 0); ghi rõ ở đây để số luồng nền không bị ẩn trong thư viện.
        if len(background) > INTERVENTIONAL_BG_MAX:
            background = shap.utils.sample(background, INTERVENTIONAL_BG_MAX, random_state=0)
        masker = shap.maskers.Independent(background, max_samples=INTERVENTIONAL_BG_MAX)
        expl = shap.TreeExplainer(model, data=masker, feature_perturbation="interventional", model_output="raw")
        if type(model).__module__.startswith("sklearn"):
            # Khi giá trị đặc trưng trùng đúng ngưỡng cắt (vd. throughput = 40000), thuật toán interventional của shap
            # đi nhánh khác với predict của scikit-learn, làm sai tính cộng ở vài luồng (đến 1e-2). Lùi mỗi ngưỡng
            # một bước số thực cho hai bên khớp nhau; kiểm tra tính cộng ở explain_stack xác nhận lại trên từng luồng.
            th = expl.model.thresholds
            expl.model.thresholds = np.where(np.isfinite(th), np.nextafter(th, -np.inf), th)
    sv = expl.shap_values(X, check_additivity=False)
    return _positive_class(sv), _base_value(expl.expected_value)


def explain_stack(models: dict, w: np.ndarray, b: float, X, background=None) -> dict:
    phis, bases, raws, err = [], [], [], {}
    for k, name in enumerate(BASE_NAMES):
        phi, base = tree_shap(models[name], X, background)
        f = raw_output(name, models[name], X)
        err[name] = np.abs(base + phi.sum(axis=1) - f)
        phis.append(phi)
        bases.append(base)
        raws.append(f)
    phi_stack = sum(w[k] * phis[k] for k in range(len(BASE_NAMES)))
    base_stack = b + sum(w[k] * bases[k] for k in range(len(BASE_NAMES)))
    z = b + sum(w[k] * raws[k] for k in range(len(BASE_NAMES)))
    err["stack"] = np.abs(base_stack + phi_stack.sum(axis=1) - z)
    return {"phi_per_base": phis, "base_per_base": bases, "raw_per_base": raws, "phi_stack": phi_stack,
            "base_stack": float(base_stack), "z": z, "additivity_error": err}


def group_attribution(phi: np.ndarray, feature_names: list[str], groups: dict[str, str], order: list[str]):
    """Cộng φ theo nhóm đặc trưng ngữ nghĩa. Trả về (ma trận n × J, danh sách nhóm)."""
    names = [g for g in order if any(groups[f] == g for f in feature_names)]
    cols = [phi[:, [i for i, f in enumerate(feature_names) if groups[f] == g]].sum(axis=1) for g in names]
    return np.stack(cols, axis=1), names


def agreement_score(group_phis: list[np.ndarray]) -> np.ndarray:
    """Điểm đồng thuận A(x): trung bình tương quan hạng (Spearman) giữa các cặp mô hình thành phần ở mức nhóm."""
    ranks = [rankdata(g, axis=1) for g in group_phis]
    cors = []
    for i in range(len(ranks)):
        for j in range(i + 1, len(ranks)):
            a = ranks[i] - ranks[i].mean(axis=1, keepdims=True)
            c = ranks[j] - ranks[j].mean(axis=1, keepdims=True)
            den = np.sqrt((a * a).sum(axis=1) * (c * c).sum(axis=1))
            cors.append(np.where(den > 0, (a * c).sum(axis=1) / np.where(den > 0, den, 1), np.nan))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(np.stack(cors, axis=1), axis=1)
