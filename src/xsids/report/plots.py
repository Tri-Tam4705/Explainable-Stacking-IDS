"""Biểu đồ so sánh dựng từ reports/*.json (và lime_sample.parquet nếu có). Mỗi hàm vẽ một hình, trả về đường dẫn.

Màu cố định để đọc nhanh: SHAP xanh dương, LIME cam, ngẫu nhiên xám; UNSW xanh ngọc, CSE tím.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from xsids.report.summaries import (
    GROUP_ORDER,
    MODEL_SHORT,
    agreement_metrics,
)

C_SHAP, C_LIME, C_RAND = "#1f77b4", "#ff7f0e", "#8c8c8c"
C_DS = {"unsw": "#2a9d8f", "cse": "#7b6fd0"}
C_SINGLE, C_VOTE, C_STACK = "#a9b8c9", "#6f8fb5", "#1f4e79"


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titlesize": 11, "axes.titleweight": "bold", "figure.dpi": 110, "savefig.dpi": 150})
    return plt


def _save(fig, out: Path, name: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    p = out / name
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return p


def _ds_with(rep: dict, key: str) -> list[str]:
    return [ds for ds in rep["datasets"] if (rep.get(key) or {}).get(ds)]


# ── 1. mô hình ───────────────────────────────────────────────────────────────
def fig_recall_per_attack(rep: dict, out: Path) -> Path | None:
    items = []
    for ds in rep["datasets"]:
        st = (rep["stacking"].get(ds) or {}).get("results", {}).get("stack_explainable")
        if st and st.get("per_attack_recall"):
            items.append((ds, st))
    if not items:
        return None
    plt = _plt()
    heights = [max(3.0, 0.38 * len(st["per_attack_recall"]) + 1.2) for _, st in items]
    fig, axes = plt.subplots(len(items), 1, figsize=(8.5, sum(heights)), squeeze=False, gridspec_kw={"height_ratios": heights})
    for (ds, st), ax in zip(items, axes[:, 0]):
        pa = sorted(st["per_attack_recall"].items(), key=lambda kv: kv[1]["recall_weighted"], reverse=True)
        names = [f"{k} (n={v['n_vectors']:,})".replace(",", ".") for k, v in pa]
        vals = [v["recall_weighted"] for _, v in pa]
        cols = ["#d62728" if v < 0.5 else "#ff9f1c" if v < 0.9 else "#2a9d8f" for v in vals]
        ax.barh(range(len(vals)), vals, color=cols)
        ax.set_yticks(range(len(vals)), names)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.12)
        for y, v in enumerate(vals):
            ax.text(v + 0.01, y, f"{v:.3f}", va="center", fontsize=8)
        ax.axvline(st["weighted"]["dr"], color="#444", ls="--", lw=1, label=f"DR chung {st['weighted']['dr']:.3f}")
        ax.legend(frameon=False, loc="lower right", fontsize=8)
        ax.set_title(f"{ds.upper()}: tỷ lệ phát hiện (recall) của stacking theo loại tấn công")
    axes[-1, 0].set_xlabel("Recall (đỏ < 0,5 · cam < 0,9 · xanh ≥ 0,9); n = số vector test của loại đó")
    fig.tight_layout()
    return _save(fig, out, "fig_recall_per_attack.png")


# ── 2. SHAP gộp ──────────────────────────────────────────────────────────────
# ── 3. LIME so với SHAP ──────────────────────────────────────────────────────
def fig_agreement(rep: dict, out: Path) -> Path | None:
    dss = _ds_with(rep, "lime")
    if not dss:
        return None
    plt = _plt()
    mets = {ds: agreement_metrics(rep["lime"][ds]) for ds in dss}
    labels = [m[0] for m in mets[dss[0]]]
    n = len(labels)
    h = 0.8 / len(dss)
    fig, ax = plt.subplots(figsize=(8.6, 0.62 * n + 1.6))
    for i, ds in enumerate(dss):
        ys = np.arange(n) + (i - (len(dss) - 1) / 2) * h
        vals = [np.nan if v is None else v for _, v in mets[ds]]
        ax.barh(ys, vals, h, color=C_DS.get(ds, "#444"), label=ds.upper())
        for y, v in zip(ys, vals):
            if not np.isnan(v):
                ax.text(v + 0.01, y, f"{v:.2f}", va="center", fontsize=8)
    ax.set_yticks(np.arange(n), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.1)
    ax.axvline(1.0, color="#bbb", lw=1)
    ax.set_xlabel("Giá trị (1,0 = hoàn hảo; riêng R² là độ khớp của mô hình thay thế LIME)")
    ax.set_title("LIME so với SHAP gộp: đồng thuận, ổn định và độ khớp")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    return _save(fig, out, "fig_agreement.png")


# ── 4. ablation, cross-dataset, LOFO ─────────────────────────────────────────
def fig_ablation(rep: dict, out: Path) -> Path | None:
    dss = _ds_with(rep, "ablation")
    if not dss:
        return None
    plt = _plt()
    cols = ["ΔDR", "ΔFAR", "ΔmacroF1", "ΔA(x)"]
    fig, axes = plt.subplots(1, len(dss), figsize=(6.4 * len(dss), 4.4), squeeze=False)
    for ax, ds in zip(axes[0], dss):
        ab = rep["ablation"][ds]
        groups = [g for g in GROUP_ORDER if g in ab["groups"]]
        M = np.array([[-ab["groups"][g]["delta"]["dr_drop"], ab["groups"][g]["delta"]["far_rise"],
                       -ab["groups"][g]["delta"]["macro_f1_drop"], (ab["agreement_ablation"].get(g) or {}).get("delta", np.nan)]
                      for g in groups], dtype=float)
        inten = np.abs(M) / np.maximum(np.nanmax(np.abs(M), axis=0, keepdims=True), 1e-12)
        ax.imshow(np.nan_to_num(inten), cmap="Reds", vmin=0, vmax=1.15, aspect="auto")
        for i in range(len(groups)):
            for j in range(4):
                v = M[i, j]
                ax.text(j, i, "-" if np.isnan(v) else (f"{v:+.4f}" if j != 1 else f"{v:+.4f}"), ha="center", va="center",
                        fontsize=8.5, color="white" if inten[i, j] > 0.6 else "#222")
        ax.set_xticks(range(4), cols)
        ax.set_yticks(range(len(groups)), groups)
        ax.set_title(f"{ds.upper()}: cắt từng nhóm đặc trưng")
        ax.spines[:].set_visible(False)
    fig.text(0.5, -0.02, "Màu đậm = thay đổi lớn nhất của cột đó (chuẩn hóa theo cột). ΔDR, ΔmacroF1 âm và ΔFAR dương là tệ đi. "
             "Cắt = thay cột bằng median (hoặc 0 với kết quả cũ), không huấn luyện lại.", ha="center", fontsize=8.5, color="#555")
    fig.tight_layout()
    return _save(fig, out, "fig_ablation.png")


def fig_cross_dataset(rep: dict, out: Path) -> Path | None:
    cl = rep.get("cross_lofo")
    if not cl or not cl.get("cross_dataset"):
        return None
    plt = _plt()
    cd = cl["cross_dataset"]
    present = {k.split("_")[1] for k in cd}
    dss = [d for d in rep["datasets"] if d in present] + sorted(present - set(rep["datasets"]))
    metrics = [("dr", "DR (cao hơn tốt hơn)", "Greens", (0, 1)), ("far", "FAR (thấp hơn tốt hơn)", "Reds", None),
               ("macro_f1", "macro-F1 (cao hơn tốt hơn)", "Greens", (0, 1))]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    for ax, (key, title, cmap, vr) in zip(axes, metrics):
        M = np.full((len(dss), len(dss)), np.nan)
        for i, tr in enumerate(dss):
            for j, te in enumerate(dss):
                r = cd.get(f"train_{tr}_test_{te}")
                if r:
                    M[i, j] = r["weighted"][key]
        vmin, vmax = vr if vr else (0, np.nanmax(M) * 1.1 if np.nanmax(M) > 0 else 1)
        ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax)
        for i in range(len(dss)):
            for j in range(len(dss)):
                if not np.isnan(M[i, j]):
                    inten = (M[i, j] - vmin) / ((vmax - vmin) or 1.0)
                    ax.text(j, i, f"{M[i, j]:.4f}" if key != "far" else f"{M[i, j]:.5f}", ha="center", va="center", fontsize=11,
                            fontweight="bold" if i == j else "normal", color="white" if inten > 0.55 else "#111")
        ax.set_xticks(range(len(dss)), [d.upper() for d in dss])
        ax.set_yticks(range(len(dss)), [d.upper() for d in dss])
        ax.set_xlabel("Kiểm tra trên")
        ax.set_ylabel("Huấn luyện trên")
        ax.set_title(title, fontsize=10)
        ax.spines[:].set_visible(False)
    fig.suptitle("Đánh giá chéo bộ dữ liệu: đường chéo (in đậm) là trong miền", fontsize=12, fontweight="bold", y=1.03)
    fig.tight_layout()
    return _save(fig, out, "fig_cross_dataset.png")


# ── tổng hợp ─────────────────────────────────────────────────────────────────
# ── 11. nhiều seed và nhiều phép chia ─────────────────────────────────────────
# (khóa, nhãn, hệ số đổi đơn vị cho chênh lệch, đơn vị, chiều tốt): FAR thấp hơn tốt hơn nên chênh lệch đảo dấu
REPEAT_METRICS = [("dr", "DR", 100.0, "điểm %", 1), ("far", "FAR", 1000.0, "điểm ‰", -1),
                  ("macro_f1", "macro-F1", 100.0, "điểm %", 1), ("pr_auc", "PR-AUC", 100.0, "điểm %", 1)]
REPEAT_MODELS = ["rf", "et", "xgb", "lgbm", "voting", "stack_explainable"]


def _repeat_ds(rep: dict) -> list[str]:
    return [ds for ds in rep["datasets"]
            if ((rep.get("multisplit") or {}).get(ds) or {}).get("metrics") and not rep["multisplit"][ds].get("fast")]


def fig_repeat_runs(rep: dict, out: Path) -> Path | None:
    """Giá trị của từng lần lặp cho 6 mô hình: chấm tròn là phép chia, tam giác là seed; vạch ngang là trung bình."""
    dss = _repeat_ds(rep)
    if not dss:
        return None
    plt = _plt()
    fig, axes = plt.subplots(len(dss), 4, figsize=(16.5, 3.5 * len(dss)), squeeze=False)
    rng = np.random.default_rng(0)
    for i, ds in enumerate(dss):
        msp = rep["multisplit"][ds]["metrics"]
        mse = ((rep.get("multiseed") or {}).get(ds) or {}).get("metrics") or {}
        n_sp = len(rep["multisplit"][ds].get("splits") or [])
        n_se = len(((rep.get("multiseed") or {}).get(ds) or {}).get("seeds") or [])
        for j, (q, lab, _, _, better) in enumerate(REPEAT_METRICS):
            ax = axes[i][j]
            scale = 1000.0 if q == "far" else 1.0
            for x, m in enumerate(REPEAT_MODELS):
                col = C_STACK if m == "stack_explainable" else C_VOTE if m == "voting" else C_SINGLE
                if m in msp:
                    v = np.array(msp[m][q]["values"]) * scale
                    ax.scatter(x - 0.13 + rng.uniform(-0.06, 0.06, len(v)), v, s=18, color=col, alpha=0.85, zorder=3,
                               edgecolor="white", linewidth=0.4)
                    ax.hlines(v.mean(), x - 0.27, x + 0.01, color=col, lw=2.2, zorder=4)
                if m in mse:
                    v = np.array(mse[m][q]["values"]) * scale
                    ax.scatter(x + 0.15 + rng.uniform(-0.05, 0.05, len(v)), v, s=20, marker="^", facecolor="white",
                               edgecolor=col, linewidth=1.0, zorder=3)
                    ax.hlines(v.mean(), x + 0.03, x + 0.27, color=col, lw=1.4, ls="--", zorder=4)
            ax.set_xticks(range(len(REPEAT_MODELS)), [MODEL_SHORT[m] for m in REPEAT_MODELS])
            ax.grid(axis="y", alpha=0.25)
            unit = " (‰)" if q == "far" else ""
            ax.set_title(f"{ds.upper()}: {lab}{unit}, {'cao' if better > 0 else 'thấp'} hơn tốt hơn", fontsize=10)
            for t in ax.get_xticklabels():
                if t.get_text() == "Stack":
                    t.set_fontweight("bold")
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], marker="o", ls="-", color="#555", label=f"phép chia train/val/test khác nhau (n = {n_sp}); vạch liền: trung bình"),
               Line2D([], [], marker="^", ls="--", color="#555", markerfacecolor="white",
                      label=f"seed huấn luyện khác nhau, cùng phép chia (n = {n_se}); vạch đứt: trung bình")]
    fig.suptitle("Kết quả từng lần lặp: mỗi điểm là một lần huấn luyện và đánh giá lại", fontsize=12.5, fontweight="bold", y=1.03)
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, out, "fig_repeat_runs.png")


def _signed_diff(entry: dict, sv: list, vv: list, better: int, factor: float, corrected: bool):
    d = (np.asarray(sv, dtype="float64") - np.asarray(vv, dtype="float64")) * better * factor
    ci = entry.get("ci95_corrected" if corrected and "ci95_corrected" in entry else "ci95") or [np.nan, np.nan]
    ci = sorted(np.asarray(ci, dtype="float64") * better * factor)
    return d, ci


def fig_repeat_diff(rep: dict, out: Path) -> Path | None:
    """Chênh lệch Stacking − Voting của từng lần lặp, đổi dấu để phía trên 0 luôn là stacking tốt hơn, kèm KTC 95%."""
    dss = _repeat_ds(rep)
    if not dss:
        return None
    plt = _plt()
    fig, axes = plt.subplots(len(dss), 4, figsize=(16.5, 3.5 * len(dss)), squeeze=False)
    rng = np.random.default_rng(1)
    for i, ds in enumerate(dss):
        srcs = []
        ms = (rep.get("multiseed") or {}).get(ds) or {}
        if ms.get("metrics") and ms.get("stack_minus_voting"):
            srcs.append(("Qua seed", ms, False, "p_value_paired_t"))
        srcs.append(("Qua phép chia", rep["multisplit"][ds], True, "p_value_corrected_t"))
        for j, (q, lab, factor, unit, better) in enumerate(REPEAT_METRICS):
            ax = axes[i][j]
            ax.axhline(0, color="#777", lw=1, ls="--", zorder=1)
            for x, (name, src, corr, pkey) in enumerate(srcs):
                met, ent = src["metrics"], src["stack_minus_voting"].get(q)
                if not ent or "stack_explainable" not in met or "voting" not in met:
                    continue
                d, ci = _signed_diff(ent, met["stack_explainable"][q]["values"], met["voting"][q]["values"], better, factor, corr)
                cols = np.where(d > 0, "#2a9d4f", np.where(d < 0, "#d1495b", "#999"))
                ax.scatter(x + rng.uniform(-0.12, 0.12, len(d)), d, s=26, c=cols, alpha=0.85, zorder=3, edgecolor="white",
                           linewidth=0.5)
                ax.errorbar(x + 0.3, d.mean(), yerr=[[d.mean() - ci[0]], [ci[1] - d.mean()]], fmt="D", color="#222",
                            ms=6, capsize=5, lw=1.6, zorder=4)
                p = ent.get(pkey, ent.get("p_value_paired_t"))
                k, n = int((d > 0).sum()), len(d)
                ptxt = "" if p is None or (isinstance(p, float) and np.isnan(p)) else f"p = {p:.3g}\n"
                ax.annotate(f"{ptxt}tốt hơn {k}/{n}", (x + 0.3, 1.0), xycoords=("data", "axes fraction"), ha="center",
                            va="top", fontsize=8, color="#333")
            ax.set_xticks(range(len(srcs)), [s[0] for s in srcs])
            ax.set_xlim(-0.5, len(srcs) - 0.2)
            lo, hi = ax.get_ylim()
            ax.set_ylim(lo, hi + (hi - lo) * 0.28)
            ax.grid(axis="y", alpha=0.25)
            ax.set_title(f"{ds.upper()}: {lab}", fontsize=10.5)
            ax.set_ylabel(f"chênh lệch ({unit})", fontsize=8.5)
    fig.suptitle("Stacking − Soft voting theo từng lần lặp (phía trên đường 0: stacking tốt hơn)\n"
                 "◆ trung bình và khoảng tin cậy 95%; p qua phép chia là kiểm định t đã hiệu chỉnh Nadeau–Bengio",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    return _save(fig, out, "fig_repeat_diff.png")


# ── 12. A(x) và cảnh báo sai; tấn công bị bỏ sót ──────────────────────────────
def fig_trust(rep: dict, out: Path, processed: Path | None) -> Path | None:
    dss = [ds for ds in rep["datasets"] if ((rep.get("trust") or {}).get(ds) or {}).get("trust")]
    if not dss or processed is None:
        return None
    import pandas as pd

    plt = _plt()
    fig, axes = plt.subplots(len(dss), 2, figsize=(12.5, 3.9 * len(dss)), squeeze=False)
    for i, ds in enumerate(dss):
        t = rep["trust"][ds]["trust"]
        p = Path(processed) / ds / "trust_sample.parquet"
        if not p.exists():
            return None
        df = pd.read_parquet(p)
        ax = axes[i][0]
        data = [df.loc[df.role == r, "agreement"].dropna().to_numpy() for r in ("TP", "FP")]
        bp = ax.boxplot(data, vert=False, widths=0.55, patch_artist=True, showfliers=False,
                        medianprops={"color": "#222", "lw": 1.8})
        for patch, c in zip(bp["boxes"], ["#2a9d4f", "#d1495b"]):
            patch.set_facecolor(c)
            patch.set_alpha(0.55)
        ax.set_yticks([1, 2], [f"Cảnh báo đúng (n={t['n_tp']})", f"Cảnh báo sai (n={t['n_fp']})"])
        ax.set_xlabel("A(x): đồng thuận giữa 4 mô hình (1 = cùng thứ tự nhóm đặc trưng)")
        ax.set_title(f"{ds.upper()}: A(x) của cảnh báo đúng và cảnh báo sai", fontsize=10.5)
        ax.grid(axis="x", alpha=0.25)
        ax = axes[i][1]
        names = ["A(x) thấp", "Độ tự tin z thấp", "A(x) thấp, trong\ncùng khoảng z"]
        rows = list(zip(names, [t["auc_low_agreement_detects_fp"], t["auc_low_z_detects_fp"], t["auc_low_agreement_within_z_quartiles"]],
                        [t["auc_low_agreement_ci95"], t["auc_low_z_ci95"], t["auc_within_z_ci95"]]))
        ys = np.arange(len(rows))
        for y, (_, v, c) in zip(ys, rows):
            ax.errorbar(v, y, xerr=[[v - c[0]], [c[1] - v]], fmt="o", color=C_STACK, capsize=4, ms=7)
            ax.text(c[1] + 0.01, y, f"{v:.2f}", va="center", fontsize=9)
        ax.axvline(0.5, color="#777", ls="--", lw=1)
        ax.set_yticks(ys, names)
        ax.invert_yaxis()
        ax.set_xlim(0.2, 1.08)
        ax.set_xlabel("ROC-AUC khi tách cảnh báo sai khỏi cảnh báo đúng (0,5 = ngẫu nhiên), KTC 95% bootstrap")
        ax.set_title(f"{ds.upper()}: A(x) có báo trước cảnh báo sai không?", fontsize=10.5)
        ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return _save(fig, out, "fig_trust.png")


def fig_missed(rep: dict, out: Path, processed: Path | None = None) -> Path | None:
    """Đóng góp SHAP gộp trung bình theo nhóm của tấn công bị bỏ sót (FN) so với tấn công bị bắt (TP).

    Mốc so sánh là các luồng bị bắt cùng loại tấn công nếu mẫu có ít nhất 10 luồng như vậy, nếu không thì là mọi luồng
    tấn công bị bắt trong mẫu (đọc từ trust_sample.parquet).
    """
    import pandas as pd

    items = []
    for ds in rep["datasets"]:
        m = (((rep.get("trust") or {}).get(ds) or {}).get("missed") or {}).get("by_attack") or {}
        for a, v in sorted(m.items(), key=lambda kv: -kv[1]["fn_weight_share"])[:2]:
            if v.get("mean_phi_tp") and v["n_tp"] >= 10:
                items.append((ds, a, v, v["mean_phi_tp"], f"bị bắt, cùng loại (n={v['n_tp']})"))
                continue
            p = None if processed is None else Path(processed) / ds / "trust_sample.parquet"
            if p is None or not p.exists():
                continue
            df = pd.read_parquet(p)
            tp = df[df.role == "TP"]
            if len(tp) < 10:
                continue
            ref = {g: float(np.average(tp[f"phi_{g}"], weights=tp.w_pop)) for g in v["mean_phi_fn"]}
            items.append((ds, a, v, ref, f"mọi tấn công bị bắt (n={len(tp)})"))
    if not items:
        return None
    plt = _plt()
    fig, axes = plt.subplots(1, len(items), figsize=(4.4 * len(items), 4.3), squeeze=False, sharey=True)
    for k, (ds, a, v, ref, ref_lab) in enumerate(items):
        ax = axes[0][k]
        gs = list(v["mean_phi_fn"])
        y = np.arange(len(gs))
        ax.barh(y - 0.2, [ref[g] for g in gs], 0.4, color="#2a9d4f", alpha=0.75, label=ref_lab)
        ax.barh(y + 0.2, [v["mean_phi_fn"][g] for g in gs], 0.4, color="#d1495b", alpha=0.85, label=f"bị bỏ sót (n={v['n_fn']})")
        ax.axvline(0, color="#555", lw=1)
        ax.set_yticks(y, gs)
        ax.invert_yaxis()
        ax.set_xlabel("φ trung bình (logit); > 0 đẩy về tấn công")
        name = a.replace("Infilteration", "Infiltration")
        ax.set_title(f"{ds.upper()}: {name}\n({v['fn_weight_share'] * 100:.1f}% số luồng bỏ sót trong mẫu)", fontsize=10)
        ax.legend(frameon=True, framealpha=0.9, fontsize=7.5, loc="best")
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle("Vì sao tấn công bị bỏ sót: đóng góp SHAP gộp trung bình theo nhóm đặc trưng", fontsize=12,
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    return _save(fig, out, "fig_missed.png")


# ── 13. kiểm tra bổ sung cho lời giải thích ──────────────────────────────────
def _ec_conditions(ec: dict, sn: dict | None) -> list[tuple[str, dict | None]]:
    """(nhãn, bộ chỉ số so sánh) cho từng kiểu thay đổi, đều đo trên SHAP gộp path-dependent trừ khi ghi khác."""
    rb = (ec.get("robust") or {}).get("levels") or {}
    sd = ((ec.get("seeds") or {}).get("path") or {}).get("feature") or {}
    seeds = {k[:-5]: v for k, v in sd.items() if k.endswith("_mean")} if sd else None
    val = (ec.get("value") or {}).get("feature") or {}
    return [("Nhiễu đầu vào ±1%", (rb.get("0.01") or {}).get("shap_path")),
            ("Nhiễu đầu vào ±5%", (rb.get("0.05") or {}).get("shap_path")),
            ("Huấn luyện lại\n(8 seed, từng cặp)", seeds),
            ("Đổi tập nền\n(interventional)", val.get("int_vs_int_other_bg")),
            ("Đổi hàm giá trị\n(path ↔ interventional)", val.get("path_vs_int")),
            ("Nhãn xáo trộn\n(kiểm tra tỉnh táo)", (sn or {}).get("feature"))]


def fig_explain_checks(rep: dict, out: Path) -> Path | None:
    """Lời giải thích thay đổi bao nhiêu dưới từng kiểu thay đổi; LIME so với SHAP khi nhiễu đầu vào."""
    dss = [ds for ds in rep["datasets"] if ((rep.get("explain_checks") or {}).get(ds) or {}) and not rep["explain_checks"][ds].get("fast")]
    if not dss:
        return None
    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), gridspec_kw={"width_ratios": [1.6, 1.6, 1.0]})
    h = 0.8 / len(dss)
    for col, (metric, title) in enumerate([("spearman_flow_mean", "Spearman |φ| trên từng luồng (1 = giữ nguyên thứ hạng)"),
                                            ("topk_jaccard_mean", "Jaccard top-5 đặc trưng trên từng luồng")]):
        ax = axes[col]
        labels = None
        for i, ds in enumerate(dss):
            conds = _ec_conditions(rep["explain_checks"][ds], (rep.get("sanity") or {}).get(ds))
            labels = [c[0] for c in conds]
            vals = [np.nan if c[1] is None else c[1].get(metric, np.nan) for c in conds]
            ys = np.arange(len(conds)) + (i - (len(dss) - 1) / 2) * h
            ax.barh(ys, vals, h, color=C_DS.get(ds, "#444"), label=ds.upper())
            for yv, v in zip(ys, vals):
                if not np.isnan(v):
                    ax.text(v + 0.01, yv, f"{v:.2f}", va="center", fontsize=8)
        ax.set_yticks(np.arange(len(labels)), labels, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.12)
        ax.axhline(len(labels) - 1.5, color="#bbb", lw=0.8, ls="--")
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x", alpha=0.25)
        if col == 0:
            ax.legend(frameon=False, loc="lower right")
    ax = axes[2]
    names = [("SHAP path, nhiễu ±1%", lambda r: r["levels"]["0.01"]["shap_path"]),
             ("SHAP interventional,\nnhiễu ±1%", lambda r: r["levels"]["0.01"]["shap_int"]),
             ("LIME, nhiễu ±1%\n(kèm seed khác)", lambda r: r["lime"]["input_and_seed"]),
             ("LIME, chỉ đổi seed", lambda r: r["lime"]["seed_only"])]
    for i, ds in enumerate(dss):
        rb = rep["explain_checks"][ds].get("robust")
        if not rb:
            continue
        vals = []
        for _, f in names:
            try:
                vals.append(f(rb)["spearman_flow_mean"])
            except (KeyError, TypeError):
                vals.append(np.nan)
        ys = np.arange(len(names)) + (i - (len(dss) - 1) / 2) * h
        ax.barh(ys, vals, h, color=C_DS.get(ds, "#444"))
        for yv, v in zip(ys, vals):
            if not np.isnan(v):
                ax.text(v + 0.01, yv, f"{v:.2f}", va="center", fontsize=8)
    ax.set_yticks(np.arange(len(names)), [n for n, _ in names], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.12)
    ax.set_title("Độ bền cục bộ: SHAP gộp và LIME\n(Spearman |φ| từng luồng)", fontsize=10)
    ax.grid(axis="x", alpha=0.25)
    fig.suptitle("Lời giải thích thay đổi bao nhiêu khi đổi đầu vào, đổi lần huấn luyện, đổi cách định nghĩa hoặc khi mô hình không học gì",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    return _save(fig, out, "fig_explain_checks.png")


def fig_deletion_knn(rep: dict, out: Path) -> Path | None:
    """Tỷ lệ luồng mất cảnh báo khi xóa k đặc trưng: thay bằng láng giềng lành tính (liền) và bằng trung vị (đứt)."""
    dss = [ds for ds in rep["datasets"] if (((rep.get("explain_checks") or {}).get(ds) or {}).get("knn")) and not rep["explain_checks"][ds].get("fast")]
    if not dss:
        return None
    plt = _plt()
    grp = [ds for ds in dss if ((rep["explain_checks"][ds]["knn"].get("full") or {}).get("group"))]
    ncol = len(dss) + (1 if grp else 0)
    fig, axes = plt.subplots(1, ncol, figsize=(4.6 * len(dss) + (4.2 if grp else 0), 4.4), squeeze=False,
                             gridspec_kw={"width_ratios": [1.1] * len(dss) + ([1.0] if grp else [])})
    sty = {"shap_path": ("SHAP gộp (path)", C_SHAP), "shap_int": ("SHAP gộp (interventional)", "#2a9d8f"),
           "lime": ("LIME", C_LIME), "random": ("Ngẫu nhiên", C_RAND)}
    if grp:  # ô cuối: xóa trọn nhóm đặc trưng, toàn mẫu giải thích
        ax = axes[0][-1]
        cats, vals = [], {"shap_path": [], "random_features": [], "random_groups": []}
        for ds in grp:
            gr = rep["explain_checks"][ds]["knn"]["full"]["group"]
            for g in sorted(gr, key=int):
                cats.append(f"{ds.upper()}\n{g} nhóm")
                for m in vals:
                    vals[m].append(gr[g][m]["flip_rate"] * 100)
        xs = np.arange(len(cats))
        bw = 0.27
        for i, (m, lab, col) in enumerate([("shap_path", "Nhóm SHAP gộp xếp cao nhất", C_SHAP),
                                           ("random_features", "Ngẫu nhiên, cùng số đặc trưng", "#9aa5ad"),
                                           ("random_groups", "Nhóm ngẫu nhiên", "#d3d8dc")]):
            bars = ax.bar(xs + (i - 1) * bw, vals[m], bw, color=col, label=lab)
            for bx, v in zip(bars, vals[m]):
                ax.text(bx.get_x() + bx.get_width() / 2, v + 1, f"{v:.0f}", ha="center", va="bottom", fontsize=7.5)
        ax.set_xticks(xs)
        ax.set_xticklabels(cats, fontsize=8.5)
        ax.set_ylim(0, 112)
        ax.set_ylabel("Luồng mất cảnh báo (%)")
        n_full = ", ".join(f"{ds.upper()} {rep['explain_checks'][ds]['knn']['full']['n_flows']}" for ds in grp)
        ax.set_title(f"Xóa trọn nhóm (láng giềng lành tính)\ntoàn mẫu giải thích:\n{n_full} luồng", fontsize=9)
        ax.grid(alpha=0.25, axis="y")
        ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    for j, ds in enumerate(dss):
        kn = rep["explain_checks"][ds]["knn"]
        ks = sorted(int(k) for k in kn["knn_fill"] if k.isdigit())
        ax = axes[0][j]
        for m, (lab, col) in sty.items():
            ax.plot([0] + ks, [0] + [kn["knn_fill"][str(k)][m]["flip_rate"] * 100 for k in ks], marker="o", color=col, label=lab)
            if m in kn["median_fill"][str(ks[0])]:
                ax.plot([0] + ks, [0] + [kn["median_fill"][str(k)][m]["flip_rate"] * 100 for k in ks], ls="--", color=col, alpha=0.55)
        ax.set_xticks([0] + ks)
        ax.set_xlabel("Số đặc trưng bị xóa (k)")
        ax.set_ylabel("Luồng mất cảnh báo (%)")
        ood = kn["knn_fill"].get("ood")
        sub = ""
        if ood:
            sub = (f"\nkhoảng cách tới dữ liệu thật: gốc {ood['nn_dist_original_median']:.2f},\n"
                   f"láng giềng {ood['nn_dist_knn_fill_median']:.2f}, trung vị {ood['nn_dist_median_fill_median']:.2f}").replace(".", ",")
        ax.set_title(f"{ds.upper()}: liền = láng giềng lành tính, đứt = trung vị{sub}", fontsize=9)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=7.8, loc="upper left")
    fig.tight_layout()
    return _save(fig, out, "fig_deletion_knn.png")


def make_all(rep: dict, out_dir, processed_dir=None, log=print) -> list[Path]:
    """Vẽ mọi hình có đủ dữ liệu. Hình nào thiếu dữ liệu hoặc lỗi thì bỏ qua và báo, không làm hỏng các hình khác."""
    out = Path(out_dir)
    jobs = [("fig_recall_per_attack", lambda: fig_recall_per_attack(rep, out)), ("fig_agreement", lambda: fig_agreement(rep, out)),
            ("fig_ablation", lambda: fig_ablation(rep, out)), ("fig_cross_dataset", lambda: fig_cross_dataset(rep, out)),
            ("fig_repeat_runs", lambda: fig_repeat_runs(rep, out)), ("fig_repeat_diff", lambda: fig_repeat_diff(rep, out)),
            ("fig_trust", lambda: fig_trust(rep, out, processed_dir)), ("fig_missed", lambda: fig_missed(rep, out, processed_dir)),
            ("fig_explain_checks", lambda: fig_explain_checks(rep, out)), ("fig_deletion_knn", lambda: fig_deletion_knn(rep, out))]
    made = []
    for name, job in jobs:
        try:
            p = job()
        except Exception as e:  # noqa: BLE001 - một hình lỗi không được làm hỏng cả bộ
            log(f"  [bỏ qua] {name}: {type(e).__name__}: {e}")
            continue
        if p is None:
            log(f"  [bỏ qua] {name}: thiếu dữ liệu")
        else:
            made.append(p)
    return made
