"""Dựng bản tóm tắt dạng chữ (bảng so sánh) từ các file JSON trong reports/.

Mỗi hàm nhận dict đọc từ JSON (hoặc dict vừa được script tạo ra, cùng cấu trúc) và trả về chuỗi,
nên các script từng bước và `scripts/13_summary.py` in ra cùng một định dạng.
"""
from __future__ import annotations

import json
from pathlib import Path

from xsids.report.console import banner, bar, best_marks, heading, num, pct, table

MODEL_ORDER = ["rf", "et", "xgb", "lgbm", "stack_explainable"]
MODEL_LABEL = {
    "rf": "Random Forest", "et": "Extra Trees", "xgb": "XGBoost", "lgbm": "LightGBM",
    "stack_explainable": "Stacking (giải thích được)",
}
MODEL_SHORT = {
    "rf": "RF", "et": "ET", "xgb": "XGB", "lgbm": "LGBM",
    "stack_explainable": "Stack",
}
GROUP_ORDER = ["volume", "timing_rate", "packet_size", "tcp_behavior", "retransmission", "protocol"]


# ── đọc dữ liệu ──────────────────────────────────────────────────────────────
def load_json(path) -> dict | None:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def load_reports(reports_dir, datasets) -> dict:
    """Đọc mọi file kết quả hiện có. Thiếu file nào thì phần đó là None và bản tóm tắt bỏ qua."""
    rdir = Path(reports_dir)
    rep: dict = {"datasets": list(datasets)}
    for name in ("baselines", "stacking", "explain", "stability", "ablation"):
        rep[name] = {ds: load_json(rdir / f"{name}_{ds}.json") for ds in datasets}
    rep["cross_lofo"] = load_json(rdir / "cross_lofo.json")
    rep["shift"] = load_json(rdir / "shift.json")                                   # tùy chọn (scripts/16_shift.py)
    rep["multiseed"] = {ds: load_json(rdir / f"multiseed_{ds}.json") for ds in datasets}  # tùy chọn (scripts/17_multiseed.py)
    rep["diagnostics"] = {ds: load_json(rdir / f"diagnostics_{ds}.json") for ds in datasets}  # tùy chọn (scripts/18_diagnostics.py)
    rep["retrain"] = {ds: load_json(rdir / f"retrain_ablation_{ds}.json") for ds in datasets}  # tùy chọn (scripts/19_retrain_ablation.py)
    rep["multisplit"] = {ds: load_json(rdir / f"multisplit_{ds}.json") for ds in datasets}  # tùy chọn (scripts/20_multisplit.py)
    rep["trust"] = {ds: load_json(rdir / f"trust_{ds}.json") for ds in datasets}  # tùy chọn (scripts/21_trust_missed.py)
    rep["explain_checks"] = {ds: load_json(rdir / f"explain_checks_{ds}.json") for ds in datasets}  # tùy chọn (scripts/22)
    rep["sanity"] = {ds: load_json(rdir / f"sanity_{ds}.json") for ds in datasets}  # tùy chọn (scripts/23)
    rep["group_shapley"] = {ds: load_json(rdir / f"group_shapley_{ds}.json") for ds in datasets}  # tùy chọn (scripts/24)
    rep["attack_type"] = {ds: load_json(rdir / f"attack_type_{ds}.json") for ds in datasets}  # tùy chọn (scripts/26)
    rep["label_audit"] = {ds: load_json(rdir / f"label_audit_{ds}.json") for ds in datasets}  # tùy chọn (scripts/27)
    return rep


def collect_models(baselines: dict | None, stacking: dict | None) -> dict:
    """Gộp kết quả mô hình đơn và stacking thành {tên: kết quả đầy đủ} theo thứ tự cố định."""
    out: dict = {}
    for src in (baselines, stacking):
        for k, v in ((src or {}).get("results") or {}).items():
            if isinstance(v, dict) and "weighted" in v:
                out[k] = v
    return {k: out[k] for k in MODEL_ORDER if k in out}


# ── mô hình ──────────────────────────────────────────────────────────────────
def models_text(ds: str, results: dict) -> str:
    """Bảng so sánh 4 mô hình đơn và stacking trên tập test. ★ = tốt nhất theo cột."""
    if not results:
        return ""
    names = list(results)
    cols = [("DR ↑", "dr", True, 4), ("FAR ↓", "far", False, 5), ("Precision ↑", "precision_at_natural_prevalence", True, 4),
            ("macro-F1 ↑", "macro_f1", True, 4), ("PR-AUC ↑", "pr_auc", True, 4)]
    vals = {key: [(results[n].get("weighted") or {}).get(key) for n in names] for _, key, _, _ in cols}
    marks = {key: best_marks(vals[key], hib) for _, key, hib, _ in cols}
    rows = []
    for i, n in enumerate(names):
        row = [MODEL_LABEL.get(n, n)]
        for _, key, _, nd in cols:
            row.append(num(vals[key][i], nd) + (" ★" if marks[key][i] else "  "))
        rows.append(row)
    return (heading(f"MÔ HÌNH — {ds.upper()} (tập test, có trọng số về tỷ lệ tấn công tự nhiên)")
            + "\n" + table(["Mô hình"] + [c[0] for c in cols], rows)
            + "\n  ★ = tốt nhất theo cột · Precision tính ở tỷ lệ tấn công tự nhiên")


def explain_text(ds: str, ex: dict) -> str:
    """SHAP gộp: số luồng, kiểm tra tính cộng, trọng số meta-learner, đóng góp theo nhóm và họ tấn công."""
    if not ex:
        return ""
    err = ex.get("additivity_error", {})
    worst = max((v.get("max", 0) for v in err.values()), default=None)
    ok = ex.get("additivity_ok")
    lines = [heading(f"SHAP GỘP — {ds.upper()}"),
             f"  Giải thích {ex.get('n_explained', '?')} trong {ex.get('n_flagged', '?')} luồng bị gắn cờ "
             f"(ngưỡng {num(ex.get('threshold'), 4)})",
             f"  Kiểm tra tính cộng: {'ĐẠT' if ok else 'KHÔNG ĐẠT'} (sai số tối đa {worst:.1e})" if worst is not None else ""]
    w = ex.get("meta_weights", {})
    if w:
        wmax = max(abs(v) for v in w.values()) or 1.0
        lines += ["", "  Trọng số meta-learner (z = b + Σ wₖ·fₖ):",
                  table(["Mô hình", "wₖ", "|wₖ| tương đối"],
                        [[MODEL_LABEL.get(k, k), num(v, 4, sign=True), bar(abs(v), wmax, 14)] for k, v in w.items()]
                        + [["Hệ số chặn b", num(ex.get("meta_bias"), 4, sign=True), ""]], "<><", indent=4)]
    fam = ex.get("mean_abs_group_attribution_by_family", {})
    if fam:
        fams = list(fam)
        groups = [g for g in GROUP_ORDER if any(g in fam[f] for f in fams)] or list(next(iter(fam.values())))
        marks = {f: best_marks([fam[f].get(g) for g in groups]) for f in fams}
        rows = [[g] + [num(fam[f].get(g), 2) + (" ★" if marks[f][i] else "  ") for f in fams] for i, g in enumerate(groups)]
        lines += ["", "  Mức đóng góp trung bình |φ| theo nhóm đặc trưng và họ tấn công (★ = nhóm lớn nhất của họ):",
                  table(["Nhóm"] + fams, rows, indent=4)]
    ag = ex.get("agreement")
    if ag:
        lines.append(f"\n  Đồng thuận A(x) giữa 4 mô hình thành phần: TB {num(ag.get('mean'), 3)} "
                     f"(p10 {num(ag.get('p10'), 3)} · p50 {num(ag.get('p50'), 3)} · p90 {num(ag.get('p90'), 3)})")
    return "\n".join(x for x in lines if x is not None)


def stability_text(ds: str, st: dict, ex: dict | None = None) -> str:
    """Độ ổn định (bootstrap) và mức đồng thuận của SHAP gộp so với hai biến thể đối chứng."""
    if not st:
        return ""
    s = st.get("stability", {})
    nb = st.get("n_bootstrap", "?")
    new_fmt = s.get("level") == "global_group_ranking"   # bản mới: thứ hạng toàn cục; bản cũ: so luồng không ghép cặp
    label = f"Kendall τ thứ hạng nhóm toàn cục ({nb} bootstrap)" if new_fmt else f"Kendall τ ({nb} bootstrap, cách tính cũ)"
    rows = [[label, f"{num(s.get('kendall_tau_mean'), 3)} ± {num(s.get('kendall_tau_std'), 3)}", bar(s.get("kendall_tau_mean"))]]
    sf = st.get("stability_feature") or {}
    if sf.get("kendall_tau_mean") is not None:
        rows.append([f"Kendall τ thứ hạng {sf.get('n_columns', '?')} đặc trưng toàn cục",
                     f"{num(sf['kendall_tau_mean'], 3)} ± {num(sf.get('kendall_tau_std'), 3)}", bar(sf["kendall_tau_mean"])])
        if sf.get("topk_jaccard_mean") is not None:
            rows.append([f"Jaccard top-{sf.get('top_k', '?')} đặc trưng giữa các lần bootstrap", num(sf["topk_jaccard_mean"], 3),
                         bar(sf["topk_jaccard_mean"])])
    rc = st.get("recompute_check") or {}
    if rc.get("per_flow_kendall_tau_mean") is not None:
        rows.append(["τ từng luồng khi tính lại SHAP (kiểm tra tất định)", num(rc["per_flow_kendall_tau_mean"], 3),
                     bar(rc["per_flow_kendall_tau_mean"])])
    if ex and ex.get("agreement"):
        rows.append(["A(x): đồng thuận 4 mô hình thành phần", num(ex["agreement"].get("mean"), 3), bar(ex["agreement"].get("mean"))])
    e1, e2 = st.get("e1_lgbm_single") or {}, st.get("e2_uniform_ensemble") or {}
    if e1.get("agreement_with_stack_mean") is not None:
        rows.append(["Đồng thuận với stacking: E1 (LightGBM đơn)", f"{num(e1['agreement_with_stack_mean'], 3)}  (p50 {num(e1.get('agreement_with_stack_p50'), 3)})",
                     bar(e1["agreement_with_stack_mean"])])
    if e2.get("agreement_with_stack_mean") is not None:
        rows.append(["Đồng thuận với stacking: E2 (gộp đều)", f"{num(e2['agreement_with_stack_mean'], 3)}  (p50 {num(e2.get('agreement_with_stack_p50'), 3)})",
                     bar(e2["agreement_with_stack_mean"])])
    r2 = (st.get("fidelity") or {}).get("r2_group_to_z")
    return (heading(f"ĐỘ ỔN ĐỊNH VÀ ĐỒNG THUẬN CỦA SHAP GỘP — {ds.upper()} ({st.get('n_sample', '?')} luồng)")
            + "\n" + table(["Chỉ số", "Giá trị", "(thang 0–1)"], rows, "<<<")
            + (f"\n  R² độ trung thực (nhóm → logit) = {num(r2, 6)}: gần như tất yếu nhờ tính cộng, không phải bằng chứng độc lập"
               if r2 is not None else "")
            + ("\n  Bootstrap = lấy mẫu lại tập luồng được giải thích rồi so thứ hạng mean |φ| với toàn bộ mẫu. TreeSHAP tất định,"
               "\n  nên trên cùng một luồng giải thích luôn giống hệt; phần đổi được chỉ là thứ hạng toàn cục theo tập luồng."
               if new_fmt else "")
            + ("\n  CẢNH BÁO: SHAP tính lại không khớp bản đã lưu ở bước 08 (mô hình hoặc mẫu đã đổi); chạy lại bước 08."
               if rc and rc.get("matches_saved") is False else ""))


def ablation_text(ds: str, ab: dict) -> str:
    """Ablation theo nhóm đặc trưng, xếp từ nhóm gây hại nhiều nhất (macro-F1 giảm nhiều nhất) đến ít nhất."""
    if not ab or not ab.get("groups"):
        return ""
    agr = ab.get("agreement_ablation", {})
    items = sorted(ab["groups"].items(), key=lambda kv: kv[1]["delta"]["macro_f1_drop"], reverse=True)
    vmax = max((abs(v["delta"]["macro_f1_drop"]) for _, v in items), default=1.0) or 1.0
    rows = []
    for rank, (g, v) in enumerate(items, 1):
        d = v["delta"]
        rows.append([rank, g, v.get("n_features_ablated", "?"), num(-d["dr_drop"], 4, True), num(d["far_rise"], 5, True),
                     num(-d["macro_f1_drop"], 4, True), bar(max(d["macro_f1_drop"], 0), vmax, 10),
                     num((agr.get(g) or {}).get("delta"), 4, True)])
    base = (ab.get("baseline") or {}).get("weighted", {})
    return (heading(f"ABLATION THEO NHÓM ĐẶC TRƯNG — {ds.upper()}")
            + f"\n  Gốc: DR {num(base.get('dr'), 4)} · FAR {num(base.get('far'), 5)} · macro-F1 {num(base.get('macro_f1'), 4)} · "
              f"A(x) {num(ab.get('agreement_baseline'), 4)}\n"
            + table(["#", "Nhóm bị cắt", "Số đ.trưng", "ΔDR", "ΔFAR", "ΔmacroF1", "Mức hại", "ΔA(x)"], rows, "><>>>><>")
            + "\n  Δ = sau khi cắt − gốc. ΔDR, ΔmacroF1 âm = tệ đi; ΔFAR dương = tệ đi. "
              + ("Cắt = thay cột bằng median của val (không huấn luyện lại, ngưỡng chọn lại trên val).\n"
                 "  ΔA(x) chỉ bỏ cột nhóm khỏi vector SHAP đã lưu, không chạy lại mô hình."
                 if ab.get("fill") == "median" else
                 "Cắt = thay cột bằng 0 (không huấn luyện lại, ngưỡng chọn lại trên val),\n"
                 "  nên ΔFAR lớn một phần do giá trị 0 lệch khỏi phân phối thật."))


def cross_text(cl: dict) -> str:
    """Đánh giá chéo bộ dữ liệu và LOFO (bỏ lần lượt từng đặc trưng)."""
    if not cl:
        return ""
    cd = cl.get("cross_dataset", {})
    rows = []
    for key, res in cd.items():
        parts = key.split("_")
        tr, te = parts[1], parts[3]
        w = res.get("weighted", {})
        ind = (cd.get(f"train_{te}_test_{te}") or {}).get("weighted", {}).get("dr")
        kept = None if tr == te or not ind else w.get("dr", 0) / ind
        rows.append([f"{tr.upper()} → {te.upper()}", "trong miền ★" if tr == te else "chéo",
                     num(w.get("dr"), 4), num(w.get("far"), 5), num(w.get("macro_f1"), 4), num(w.get("pr_auc"), 4),
                     "-" if kept is None else pct(kept, 1), bar(w.get("dr"), 1.0, 10)])
    out = [heading("ĐÁNH GIÁ CHÉO BỘ DỮ LIỆU (huấn luyện → kiểm tra)"),
           table(["Huấn luyện → Kiểm tra", "Loại", "DR", "FAR", "macro-F1", "PR-AUC", "DR so với trong miền", "DR"], rows, "<<>>>>>>"),
           "  DR chéo gần 0 nghĩa là mô hình huấn luyện ở một bộ gần như không phát hiện được tấn công ở bộ kia."]
    lofo = cl.get("lofo", {})
    if lofo:
        dss = list(lofo)
        top = {ds: list(lofo[ds]["features"].items())[:10] for ds in dss}
        head = ["#"]
        for ds in dss:
            head += [f"{ds.upper()}: đặc trưng (nhóm)", "ΔF1"]
        rows = []
        for i in range(10):
            r = [i + 1]
            for ds in dss:
                if i < len(top[ds]):
                    f, v = top[ds][i]
                    r += [f"{f} ({v.get('group', '?')})", num(v["f1_drop"], 4)]
                else:
                    r += ["", ""]
            rows.append(r)
        base = "  ·  ".join(f"{ds.upper()} macro-F1 gốc {num(lofo[ds].get('baseline_macro_f1'), 4)}" for ds in dss)
        out += [heading("LOFO — 10 đặc trưng làm macro-F1 giảm nhiều nhất khi bị loại"), "  " + base,
                table(head, rows, ">" + "<>" * len(dss)),
                ("  Loại = thay cột bằng median của val, không huấn luyện lại, chọn lại ngưỡng trên val."
                 if cl.get("fill") == "median" else
                 "  Loại = thay cột bằng 0, không huấn luyện lại, chọn lại ngưỡng trên val. Mức giảm rất lớn ở một đặc trưng đơn lẻ nên được kiểm tra lại.")]
    return "\n".join(out)


def shift_text(sh: dict, top: int = 10) -> str:
    """Dịch chuyển phân phối giữa hai bộ dữ liệu và đánh giá chéo theo xếp hạng/ngưỡng (scripts/16_shift.py)."""
    if not sh:
        return ""
    out = []
    feats = sh.get("features") or []
    if feats:
        rows = [[r["feature"], num(r.get("ks_all"), 3), num(r.get("ks_benign"), 3), num(r.get("psi_all"), 2),
                 num(r.get("psi_benign"), 2)] for r in feats[:top]]
        n_big = sum((r.get("psi_benign") or 0) > 0.25 for r in feats)
        out += [heading(f"DỊCH CHUYỂN PHÂN PHỐI GIỮA UNSW VÀ CSE — {top} đặc trưng lệch nhất (tập val, xếp theo PSI luồng lành tính)"),
                table(["Đặc trưng", "KS tất cả", "KS lành tính", "PSI tất cả", "PSI lành tính"], rows),
                f"  {n_big}/{len(feats)} đặc trưng có PSI (lành tính) > 0,25, mức thường coi là lệch lớn. Lệch ngay ở luồng lành tính"
                "\n  nghĩa là hai môi trường mạng khác nhau, không chỉ khác loại tấn công."]
    cross = sh.get("cross") or {}
    if cross:
        rows = []
        for k, v in cross.items():
            tr, te = k.split("->")
            s, c = v.get("source_threshold", {}), v.get("recalibrated_threshold", {})
            rows.append([f"{tr.upper()} → {te.upper()}", pct(v.get("prevalence"), 2), num(v.get("roc_auc"), 3), num(v.get("pr_auc"), 3),
                         f"{num(s.get('dr'), 3)} / {num(s.get('far'), 5)}", f"{num(c.get('dr'), 3)} / {num(c.get('far'), 5)}"])
        out += [heading("ĐÁNH GIÁ CHÉO: XẾP HẠNG VÀ NGƯỠNG (tập test đích, có trọng số)"),
                table(["Huấn luyện → Kiểm tra", "Tỷ lệ TC", "ROC-AUC", "PR-AUC", "Ngưỡng gốc DR / FAR", "Ngưỡng chỉnh lại DR / FAR"],
                      rows, "<>>>>>"),
                "  ROC-AUC < 0,5: thứ tự điểm số bị đảo ở bộ đích. Ngưỡng chỉnh lại = ngưỡng F1 tốt nhất trên val của bộ đích (cần nhãn đích)."
                "\n  Nếu nó cho DR thấp, hoặc DR ≈ 1 kèm FAR ≈ 1 (gắn cờ mọi luồng), thì không ngưỡng nào tách được tấn công:"
                "\n  nguyên nhân là phân phối đặc trưng chứ không phải ngưỡng."]
    return "\n".join(out)


def _stack_vs_base_rows(paired: dict, mets, corrected: bool) -> list[str]:
    """Bảng chênh lệch Stacking − từng mô hình đơn (mỗi mô hình một bảng nhỏ)."""
    out = []
    for base, ent in paired.items():
        rows = []
        for m, lab, nd in mets:
            if m not in ent:
                continue
            p = ent[m]
            row = [lab, f"{num(p['mean_diff'], nd, True)} ± {num(p['std_diff'], nd)}", num(p.get("p_value_paired_t"), 4)]
            if corrected:
                row.append(num(p.get("p_value_corrected_t"), 4))
            row += [num(p.get("p_value_wilcoxon"), 4), f"{p['a_better_in']}/{p['n']}"]
            rows.append(row)
        head = ["Chỉ số", "Chênh lệch TB ± ĐLC", "p (t)"] + (["p (t hiệu chỉnh)"] if corrected else []) + ["p (Wilcoxon)", "Stacking tốt hơn ở"]
        out += ["", f"  Stacking (giải thích được) − {MODEL_LABEL.get(base, base)}:",
                table(head, rows, "<" + ">" * (len(head) - 1), indent=4)]
    return out


def multiseed_text(ds: str, ms: dict) -> str:
    """Trung bình ± độ lệch chuẩn qua nhiều seed huấn luyện và chênh lệch ghép cặp Stacking − từng mô hình đơn (scripts/17_multiseed.py)."""
    if not ms or not ms.get("metrics"):
        return ""
    mets = [("dr", "DR", 4), ("far", "FAR", 5), ("macro_f1", "macro-F1", 4), ("pr_auc", "PR-AUC", 4)]
    rows = [[MODEL_LABEL.get(k, k)] + [f"{num(v[m]['mean'], nd)} ± {num(v[m]['std'], nd)}" for m, _, nd in mets if m in v]
            for k, v in ms["metrics"].items()]
    seeds = ms.get("seeds") or []
    out = [heading(f"NHIỀU SEED — {ds.upper()}: trung bình ± độ lệch chuẩn qua {len(seeds)} lần huấn luyện ({', '.join(map(str, seeds))})"),
           table(["Mô hình"] + [lab for _, lab, _ in mets], rows)]
    paired = ms.get("stack_vs_base") or {}
    if paired:
        out += _stack_vs_base_rows(paired, mets, corrected=False)
        out += ["  |chênh lệch| nhỏ hơn độ lệch chuẩn: chưa đủ cơ sở xếp hạng hai mô hình. Độ lệch chuẩn chỉ phản ánh"
                "\n  tính ngẫu nhiên của huấn luyện (phép chia train/val/test giữ nguyên)."]
    return "\n".join(out)


def multisplit_text(ds: str, ms: dict) -> str:
    """Trung bình ± độ lệch chuẩn qua nhiều phép chia train/val/test và kiểm định t ghép cặp (scripts/20_multisplit.py)."""
    if not ms or not ms.get("metrics") or ms.get("fast"):
        return ""
    mets = [("dr", "DR", 4), ("far", "FAR", 5), ("macro_f1", "macro-F1", 4), ("pr_auc", "PR-AUC", 4)]
    rows = [[MODEL_LABEL.get(k, k)] + [f"{num(v[m]['mean'], nd)} ± {num(v[m]['std'], nd)}" for m, _, nd in mets]
            for k, v in ms["metrics"].items()]
    splits = ms.get("splits") or []
    out = [heading(f"NHIỀU PHÉP CHIA — {ds.upper()}: trung bình ± độ lệch chuẩn qua {len(splits)} phép chia "
                   f"train/val/test (seed chia {', '.join(map(str, splits))})"),
           table(["Mô hình"] + [lab for _, lab, _ in mets], rows)]
    paired = ms.get("stack_vs_base") or {}
    if paired:
        out += ["  t hiệu chỉnh: Nadeau-Bengio; kiểm định hai phía."] + _stack_vs_base_rows(paired, mets, corrected=True)
        out += ["  Độ lệch chuẩn gồm cả ngẫu nhiên của việc chia dữ liệu. 25 đặc trưng giữ nguyên ở mọi phép chia."]
    return "\n".join(out)


def trust_text(ds: str, tr: dict) -> str:
    """A(x) và cảnh báo sai; đóng góp SHAP gộp của tấn công bị bỏ sót (scripts/21_trust_missed.py)."""
    if not tr:
        return ""
    out = [heading(f"A(x) VÀ CẢNH BÁO SAI; TẤN CÔNG BỊ BỎ SÓT — {ds.upper()}")]
    t = tr.get("trust") or {}
    if t:
        ci = lambda c: f"[{num(c[0], 3)}; {num(c[1], 3)}]"  # noqa: E731
        out += [f"  Mẫu: {t['n_fp']} cảnh báo sai, {t['n_tp']} cảnh báo đúng (trọng số khôi phục tỷ lệ của tập test).",
                f"  ROC-AUC tách cảnh báo sai bằng A(x) thấp: {num(t['auc_low_agreement_detects_fp'], 3)} {ci(t['auc_low_agreement_ci95'])}",
                f"  ROC-AUC tách cảnh báo sai bằng z thấp:    {num(t['auc_low_z_detects_fp'], 3)} {ci(t['auc_low_z_ci95'])}",
                f"  ROC-AUC của A(x) trong cùng khoảng z:    {num(t['auc_low_agreement_within_z_quartiles'], 3)} {ci(t['auc_within_z_ci95'])}",
                "  Tỷ lệ cảnh báo sai theo tứ phân vị A(x) (thấp → cao): "
                + ", ".join(pct(v) for v in t["fp_share_by_agreement_quartile"])]
    m = (tr.get("missed") or {}).get("by_attack") or {}
    for a, v in sorted(m.items(), key=lambda kv: -kv[1]["fn_weight_share"]):
        low = sorted(v["mean_phi_fn"].items(), key=lambda kv: kv[1])[:2]
        out.append(f"  {a}: {v['n_fn']} luồng bỏ sót ({pct(v['fn_weight_share'])} số luồng bỏ sót), z TB {num(v['mean_z_fn'], 2)} "
                   f"(ngưỡng {num(tr['missed']['z_threshold'], 2)}); nhóm kéo về lành tính: "
                   + ", ".join(f"{g} {num(x, 2, True)}" for g, x in low))
    return "\n".join(out)


def explain_checks_text(ds: str, ec: dict, sn: dict | None) -> str:
    """Kiểm tra bổ sung cho lời giải thích (scripts/22_explain_checks.py, 23_sanity_randomization.py)."""
    if not ec or ec.get("fast"):
        return ""
    out = [heading(f"KIỂM TRA BỔ SUNG CHO LỜI GIẢI THÍCH — {ds.upper()}")]
    v = ec.get("value")
    if v:
        f, g = v["feature"], v["group"]
        out += [f"  Hàm giá trị: SHAP interventional (nền {ec.get('background_size')} luồng lành tính), sai số tính cộng lớn nhất "
                f"{max(v['additivity_error_interventional'].values()):.1e}",
                f"    path ↔ interventional: Spearman từng luồng {num(f['path_vs_int']['spearman_flow_mean'], 3)}, "
                f"top-5 Jaccard {num(f['path_vs_int']['topk_jaccard_mean'], 3)}, τ nhóm toàn cục {num(g['path_vs_int']['global_kendall'], 3)}",
                f"    hai tập nền khác nhau: Spearman {num(f['int_vs_int_other_bg']['spearman_flow_mean'], 3)}, "
                f"τ nhóm {num(g['int_vs_int_other_bg']['global_kendall'], 3)}"]
    kn = ec.get("knn")
    if kn:
        rows = []
        for k in sorted(int(x) for x in kn["knn_fill"] if x.isdigit()):
            for m, lab in (("shap_path", "SHAP path"), ("shap_int", "SHAP interventional"), ("random", "Ngẫu nhiên")):
                rows.append([str(k), lab, pct(kn["knn_fill"][str(k)][m]["flip_rate"]), pct(kn["median_fill"][str(k)][m]["flip_rate"])])
        out += ["", "  Phép xóa trong phân phối (tỷ lệ luồng mất cảnh báo):",
                table(["k", "Phương pháp", "Thay bằng láng giềng", "Thay bằng trung vị"], rows, "<<>>", indent=4)]
        o = kn["knn_fill"].get("ood")
        if o:
            out.append(f"    Khoảng cách tới luồng thật gần nhất (trung vị): gốc {num(o['nn_dist_original_median'], 2)}, "
                       f"thay bằng láng giềng {num(o['nn_dist_knn_fill_median'], 2)}, thay bằng trung vị {num(o['nn_dist_median_fill_median'], 2)}")

        def ci(d):
            c = d.get("ci95")
            return f"{num(d['mean_diff'], 3)} [{num(c[0], 3)}; {num(c[1], 3)}]" if c else num(d["mean_diff"], 3)

        for tag, block in (("mẫu con", kn), ("toàn mẫu giải thích", kn.get("full") or {})):
            gr = block.get("group")
            if not gr:
                continue
            n_txt = f", {block['n_flows']} luồng" if "n_flows" in block else ""
            out.append(f"  Xóa trọn nhóm đặc trưng, thay bằng láng giềng ({tag}{n_txt}):")
            for g in sorted(gr, key=int):
                r = gr[g]
                meths = [m for m in ("shap_path", "shap_int") if m in r]
                parts = ", ".join(f"{m} {pct(r[m]['flip_rate'])}" for m in meths)
                line = (f"    {g} nhóm: mất cảnh báo {parts}; ngẫu nhiên cùng số đặc trưng {pct(r['random_features']['flip_rate'])}, "
                        f"nhóm ngẫu nhiên {pct(r['random_groups']['flip_rate'])}")
                if "vs_random_features" in r["shap_path"]:
                    line += f" | chênh lệch mức giảm xác suất SHAP path − ngẫu nhiên cùng số đặc trưng: {ci(r['shap_path']['vs_random_features'])}"
                out.append(line)
            kf = block.get("knn_fill") if tag != "mẫu con" else None
            if kf:
                ks_ = sorted(int(x) for x in kf if x.isdigit())
                out.append("    Xóa k đặc trưng (SHAP path / ngẫu nhiên): " + ", ".join(
                    f"k={k}: {pct(kf[str(k)]['shap_path']['flip_rate'])} / {pct(kf[str(k)]['random']['flip_rate'])}" for k in ks_))
    rb = ec.get("robust")
    if rb:
        out.append("")
        for rel, lv in rb["levels"].items():
            out.append(f"  Nhiễu ±{float(rel) * 100:.0f}%: SHAP path Spearman {num(lv['shap_path']['spearman_flow_mean'], 3)}, "
                       f"top-5 Jaccard {num(lv['shap_path']['topk_jaccard_mean'], 3)}; SHAP interventional Spearman "
                       f"{num(lv['shap_int']['spearman_flow_mean'], 3)}; |Δz| trung vị {num(lv['median_abs_dz'], 3)}")
    sd = ec.get("seeds")
    if sd:
        pf, pg = sd["path"]["feature"], sd["path"]["group"]
        out += ["", f"  Qua {len(sd['models'])} mô hình (seed): τ nhóm toàn cục {num(pg['global_kendall_mean'], 3)} (nhỏ nhất "
                f"{num(pg['global_kendall_min'], 3)}), Spearman từng luồng {num(pf['spearman_flow_mean_mean'], 3)}, top-5 Jaccard "
                f"{num(pf['topk_jaccard_mean_mean'], 3)}, trùng nhóm hạng 1 {pct(pg['top1_same_mean'])}"]
    if sn and not sn.get("fast"):
        out.append(f"  Nhãn xáo trộn (ROC-AUC {num(sn['roc_auc_random_labels'], 3)}): Spearman từng luồng với mô hình thật "
                   f"{num(sn['feature']['spearman_flow_mean'], 3)}, top-5 Jaccard {num(sn['feature']['topk_jaccard_mean'], 3)}, "
                   f"trùng nhóm hạng 1 {pct(sn['group']['top1_same'])}")
    wg = ec.get("weighted")
    if wg:
        out.append(f"  Thứ hạng nhóm có trọng số theo dòng cảnh báo thật: {', '.join(wg['rank_weighted'])} (τ với không trọng số "
                   f"{num(wg['kendall'], 3)})")
    return "\n".join(out)


GS_LABELS = {"shap_path": "SHAP gộp (path)", "shap_int": "SHAP gộp (interventional)", "group_shapley": "Shapley nhóm",
             "oracle": "Oracle (mức tối đa)", "random": "Ngẫu nhiên (kỳ vọng)"}


def group_shapley_text(ds: str, gs: dict) -> str:
    """Shapley nhóm trong phân phối: độ trung thực so với oracle (scripts/24_group_shapley.py)."""
    if not gs or gs.get("fast"):
        return ""
    out = [heading(f"SHAPLEY NHÓM TRONG PHÂN PHỐI — {ds.upper()}")]
    for tag, block in (("mẫu con", gs.get("group")), ("toàn mẫu giải thích", gs.get("full"))):
        if not block:
            continue
        dl, fc = block["deletion"], block["faithfulness_correlation"]
        gks = sorted((k for k in dl if k.isdigit()), key=int)
        meths = [m for m in GS_LABELS if m in dl[gks[0]]]
        rows = [[GS_LABELS[m]] + [pct(dl[g][m]["flip_rate"], 1) for g in gks]
                + [num(fc[m]["mean"], 2) if m in fc else "-"] for m in meths]
        out += [f"  Xóa g nhóm trong phân phối ({tag}, {dl['n_flows']} luồng): tỷ lệ mất cảnh báo và tương quan trung thực",
                table(["Phương pháp"] + [f"g={g}" for g in gks] + ["Tương quan"], rows, "<" + ">" * (len(gks) + 1), indent=4),
                f"    Trần của tương quan (Shapley nhóm tính bằng chính phép đo): {num(fc['ceiling_same_operator']['mean'], 2)}"]
    g = gs.get("group")
    if g:
        s = g["stability_ref_B_vs_C"]
        out.append(f"  Đổi tập tham chiếu lành tính: Spearman {num(s['spearman_signed_mean'], 2)}, trùng hạng 1 "
                   f"{pct(s['top1_signed_same'])}; {num(g['seconds_per_flow_one_table'], 2)} s mỗi luồng")
        if "kernel_check_max_abs_diff" in g:
            out.append(f"  Kiểm tra chéo bằng kernel Shapley trên đủ liên minh: sai lệch lớn nhất {g['kernel_check_max_abs_diff']:.1e}")
    fe = gs.get("feature")
    if fe:
        v, sp = fe["kernelshap_vs_exact_interventional"], fe["seconds_per_flow"]
        out.append(f"  Mức đặc trưng ({fe['n_flows']} luồng, {fe['n_coalitions']} liên minh): KernelSHAP so với SHAP gộp "
                   f"interventional chính xác, Spearman {num(v['spearman_signed_mean'], 2)}; thời gian mỗi luồng: SHAP gộp "
                   f"chính xác {num(sp['shap_stack_exact_interventional'], 3)} s, KernelSHAP {num(sp['sampled_kernelshap'], 2)} s")
    return "\n".join(out)


def attack_type_text(ds: str, at: dict | None) -> str:
    """Bước hai đa lớp gọi tên loại tấn công (scripts/26_attack_type.py)."""
    if not at or at.get("fast"):
        return ""
    t, e = at["test"], at["end_to_end"]["overall"]
    rows = [[c, f"{v['n_weighted']:,}", num(v["precision"], 3), num(v["recall"], 3), num(v["f1"], 3),
             at["explain"].get(c, {}).get("top_group", "-")] for c, v in t["per_class"].items() if v["n_vectors"]]
    return "\n".join([
        heading(f"GỌI TÊN LOẠI TẤN CÔNG (BƯỚC HAI ĐA LỚP) — {ds.upper()}"),
        f"  Cấu hình chọn trên val: {at['config']}",
        f"  Test (vector tấn công): độ chính xác {num(t['accuracy'], 4)}, macro-F1 {num(t['macro_f1'], 4)}; "
        f"trần do vector trùng nhiều tên {num(at['ceiling']['accuracy_ceiling'], 4)}",
        f"  Đầu cuối: gắn cờ {pct(e['detected'], 1)}, gắn cờ và gọi đúng tên {pct(e['detected_and_named'], 1)} "
        f"({pct(e['named_given_detected'], 1)} trong số luồng bị gắn cờ)",
        table(["Loại", "Luồng test", "Precision", "Recall", "F1", "Nhóm SHAP lớn nhất"], rows, "<>>>><", indent=4)])


def label_audit_text(ds: str, la: dict | None) -> str:
    """Kiểm toán nhãn: trùng vector khác nhãn, học tự tin, ảnh hưởng (scripts/27_label_audit.py)."""
    if not la or la.get("fast"):
        return ""
    ov, cl, ef = la["overlap"], la["confident_learning"], la["effect"]
    att = {c: v for c, v in ov["by_attack"].items() if v["label"] == 1}
    rows = [[c, f"{v['n_weighted']:,}", pct(v["shared_with_other_label"], 1), pct(v["unreachable"], 1),
             pct(cl["by_attack"][c]["suspect_share_weighted"], 1)] for c, v in att.items()]
    b = cl["by_attack"].get("Benign", {})
    out = [heading(f"KIỂM TOÁN NHÃN — {ds.upper()}"),
           f"  Trần recall do trùng 25 đặc trưng với luồng Benign chiếm đa số: {pct(ov['overall']['attack_recall_ceiling'], 2)}",
           f"  Học tự tin (OOF LightGBM): nghi sai nhãn {pct(cl['suspect_share_vectors'], 2)} vector, "
           f"{pct(cl['suspect_share_weighted'], 2)} luồng; Benign nghi sai {pct(b.get('suspect_share_weighted', 0), 3)}",
           table(["Loại", "Luồng", "Trùng với nhãn khác", "Không thể bắt", "Nghi sai nhãn"], rows, "<>>>>", indent=4)]
    m0, m1 = ef["main_model"]["test_all"]["weighted"], ef["main_model"]["test_without_suspects"]["weighted"]
    out.append(f"  Mô hình chính: DR {num(m0['dr'], 4)} -> {num(m1['dr'], 4)}, FAR {num(m0['far'], 6)} -> {num(m1['far'], 6)} "
               f"khi bỏ {ef['n_test_suspects']:,} vector test nghi sai nhãn (cận trên)")
    rt = ef.get("retrained_without_suspects")
    if rt:
        a, c = rt["test_all"]["weighted"], rt["test_without_suspects"]["weighted"]
        out.append(f"  Huấn luyện lại không có vector nghi sai: test đầy đủ DR {num(a['dr'], 4)}, FAR {num(a['far'], 6)}, "
                   f"PR-AUC {num(a['pr_auc'], 4)}; test đã bỏ vector nghi sai DR {num(c['dr'], 4)}, FAR {num(c['far'], 6)}")
    return "\n".join(out)


def retrain_text(ds: str, rt: dict) -> str:
    """Ablation có huấn luyện lại: bỏ nhóm đặc trưng rồi huấn luyện lại cả hệ (scripts/19_retrain_ablation.py)."""
    if not rt or not rt.get("configs") or rt.get("fast"):
        return ""
    cfgs = rt["configs"]
    base = (cfgs.get("full") or {}).get("in_domain", {}).get("stack_explainable")
    rows = []
    for name, c in cfgs.items():
        s, x = c["in_domain"]["stack_explainable"], c["cross"]["stack_explainable"]
        d = num(s["macro_f1"] - base["macro_f1"], 4, True) if base and name != "full" else "-"
        rows.append([name, str(c["n_kept"]), num(s["dr"], 4), num(s["far"], 5), num(s["macro_f1"], 4), num(s["pr_auc"], 4), d,
                     num(x["dr"], 4), num(x["far"], 5), num(x["roc_auc"], 3)])
    return "\n".join([
        heading(f"ABLATION CÓ HUẤN LUYỆN LẠI — {ds.upper()} (stacking; chéo = kiểm tra trên bộ kia với ngưỡng nguồn)"),
        table(["Cấu hình", "Số đ.trưng", "DR", "FAR", "macro-F1", "PR-AUC", "Δ macro-F1", "Chéo DR", "Chéo FAR", "Chéo ROC-AUC"], rows),
        "  Mô hình được huấn luyện lại không có các đặc trưng bị bỏ; ngưỡng chọn lại trên val."])


def missed_text(ds: str, dg: dict) -> str:
    """Tấn công bị bỏ sót theo loại (phần C của scripts/18_diagnostics.py)."""
    m = (dg or {}).get("missed")
    if not m or dg.get("fast"):
        return ""
    rows = [[r["attack"], str(r["n_vectors"]), f"{r['n_weighted']:.0f}", num(r["recall"], 4), f"{r['missed_weighted']:.0f}",
             pct(r["missed_share_of_attack_mass"], 2), pct(r["missed_also_by_all_base_models"]),
             pct(r["missed_in_label_conflict_groups"])]
            for r in m["by_attack"] if r["missed_weighted"] >= 0.5]
    out = [heading(f"TẤN CÔNG BỊ BỎ SÓT — {ds.upper()} (stacking, DR {num(m['dr'], 4)}; "
                   f"{m['missed_mass']:.0f} trong {m['attack_mass']:.0f} luồng tấn công bị bỏ sót)"),
           table(["Loại tấn công", "vector", "luồng", "recall", "bỏ sót", "% tổng TC", "cả 4 mô hình bỏ sót", "nhóm xung đột nhãn"], rows)]
    d = m.get("dr_without_top_attack")
    if d:
        out.append(f"  DR nếu không tính loại '{d['attack']}': {num(d['dr'], 4)}")
    return "\n".join(out)


# ── so sánh hai bộ dữ liệu ───────────────────────────────────────────────────
def compare_text(rep: dict) -> str:
    """Bảng đối chiếu các chỉ số chính giữa các bộ dữ liệu, mỗi dataset một cột."""
    dss = [ds for ds in rep["datasets"] if (rep.get("stacking") or {}).get(ds)]
    if not dss:
        return ""
    rows: list[list] = []

    def add(label, fn, nd=3, as_pct=False):
        vals = []
        for ds in dss:
            try:
                v = fn(ds)
            except (KeyError, TypeError, IndexError, AttributeError):
                v = None
            vals.append(pct(v, 1) if as_pct else num(v, nd))
        rows.append([label] + vals)

    stack = {ds: (collect_models(rep["baselines"].get(ds), rep["stacking"].get(ds)).get("stack_explainable") or {}).get("weighted", {})
             for ds in dss}
    rows.append(["MÔ HÌNH STACKING"])
    add("DR", lambda ds: stack[ds]["dr"], 4)
    add("FAR", lambda ds: stack[ds]["far"], 5)
    add("macro-F1", lambda ds: stack[ds]["macro_f1"], 4)
    rows.append(["SHAP GỘP"])
    add("Kendall τ thứ hạng nhóm (bootstrap)", lambda ds: rep["stability"][ds]["stability"]["kendall_tau_mean"])
    add("Kendall τ thứ hạng đặc trưng (bootstrap)", lambda ds: rep["stability"][ds]["stability_feature"]["kendall_tau_mean"])
    add("A(x) đồng thuận 4 mô hình", lambda ds: rep["explain"][ds]["agreement"]["mean"])
    gsf = {ds: (((rep.get("group_shapley") or {}).get(ds) or {}).get("full") or {}).get("deletion", {}).get("1", {}) for ds in dss}
    if any(gsf.values()):
        rows.append(["TRUNG THỰC: XÓA 1 NHÓM TRONG PHÂN PHỐI (toàn mẫu giải thích)"])
        add("Mất cảnh báo — SHAP gộp", lambda ds: gsf[ds]["shap_path"]["flip_rate"], as_pct=True)
        add("Mất cảnh báo — Shapley nhóm", lambda ds: gsf[ds]["group_shapley"]["flip_rate"], as_pct=True)
        add("Mất cảnh báo — oracle", lambda ds: gsf[ds]["oracle"]["flip_rate"], as_pct=True)
        add("Mất cảnh báo — ngẫu nhiên", lambda ds: gsf[ds]["random"]["flip_rate"], as_pct=True)
    return heading("SO SÁNH GIỮA CÁC BỘ DỮ LIỆU — các chỉ số chính") + "\n" + table(["Chỉ số"] + [d.upper() for d in dss], rows)


def render_all(rep: dict) -> str:
    """Toàn bộ bản tóm tắt: mô hình, SHAP, ổn định, ablation, cross/LOFO và bảng so sánh."""
    parts = [banner("TỔNG HỢP KẾT QUẢ HỆ THỐNG EXPLAINABLE STACKING IDS")]
    for ds in rep["datasets"]:
        parts.append(models_text(ds, collect_models(rep["baselines"].get(ds), rep["stacking"].get(ds))))
    for ds in rep["datasets"]:
        parts.append(multiseed_text(ds, (rep.get("multiseed") or {}).get(ds)))
    for ds in rep["datasets"]:
        parts.append(multisplit_text(ds, (rep.get("multisplit") or {}).get(ds)))
    for ds in rep["datasets"]:
        parts.append(missed_text(ds, (rep.get("diagnostics") or {}).get(ds)))
        parts.append(trust_text(ds, (rep.get("trust") or {}).get(ds)))
        parts.append(explain_checks_text(ds, (rep.get("explain_checks") or {}).get(ds), (rep.get("sanity") or {}).get(ds)))
        parts.append(group_shapley_text(ds, (rep.get("group_shapley") or {}).get(ds)))
        parts.append(attack_type_text(ds, (rep.get("attack_type") or {}).get(ds)))
        parts.append(label_audit_text(ds, (rep.get("label_audit") or {}).get(ds)))
    for ds in rep["datasets"]:
        parts.append(explain_text(ds, rep["explain"].get(ds)))
    for ds in rep["datasets"]:
        parts.append(stability_text(ds, rep["stability"].get(ds), rep["explain"].get(ds)))
    for ds in rep["datasets"]:
        parts.append(ablation_text(ds, rep["ablation"].get(ds)))
    for ds in rep["datasets"]:
        parts.append(retrain_text(ds, (rep.get("retrain") or {}).get(ds)))
    parts.append(cross_text(rep.get("cross_lofo")))
    parts.append(shift_text(rep.get("shift")))
    parts.append(compare_text(rep))
    return "\n".join(p for p in parts if p) + "\n"
