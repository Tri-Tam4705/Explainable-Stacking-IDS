"""Kiểm toán dữ liệu thô: lỗi dữ liệu, trùng lặp, lối tắt, tương quan.

Chạy trên dữ liệu SAU khi nạp nhưng TRƯỚC khi làm sạch, nên còn cổng và TTL để chứng minh
chúng là đặc trưng "lối tắt" (AUC đơn đặc trưng rất cao).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from xsids.data.cleaning import feature_columns, label_conflicts, nonfinite_mask
from xsids.schema import ALL_FEATURES, ATTACK_COL, FAMILY_COL, HARD_DROP_COLS, LABEL_COL


def single_feature_auc(df: pd.DataFrame, cols: list[str], sample_n: int, seed: int = 0) -> dict:
    s = df.sample(n=min(sample_n, len(df)), random_state=seed)
    y = s[LABEL_COL].to_numpy()
    if y.min() == y.max():
        return {}
    out = {}
    for c in cols:
        x = s[c].to_numpy(dtype="float64", na_value=np.nan)
        ok = np.isfinite(x)
        if not ok.any() or x[ok].min() == x[ok].max():
            out[c] = 0.5
            continue
        x = np.where(ok, x, np.median(x[ok]))
        auc = roc_auc_score(y, x)
        out[c] = float(max(auc, 1.0 - auc))  # không phân biệt chiều
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def value_ranges(df: pd.DataFrame, cols: list[str], sample_n: int, abs_thr: float, seed: int = 0):
    """min / trung vị / p99 / max từng đặc trưng và số giá trị vượt ngưỡng vật lý (giá trị đệm)."""
    rng = np.random.default_rng(seed)
    out, implausible = {}, {}
    for c in cols:
        x = df[c].to_numpy(dtype="float64", na_value=np.nan)
        xf = x[np.isfinite(x)]
        if len(xf) == 0:
            continue
        xs = xf if len(xf) <= sample_n else xf[rng.choice(len(xf), sample_n, replace=False)]
        out[c] = {"min": float(xf.min()), "p50": float(np.median(xs)),
                  "p99": float(np.quantile(xs, 0.99)), "max": float(xf.max())}
        n_big = int((np.abs(xf) > abs_thr).sum())
        if n_big:
            implausible[c] = n_big
    return out, implausible


def saturation_report(df: pd.DataFrame, cols: list[str], min_share: float = 0.001, tol: float = 0.999):
    """Đặc trưng có >= min_share số dòng nằm sát giá trị lớn nhất (bão hòa, tràn số, giá trị đệm).

    Kèm tỷ lệ tấn công trong nhóm đó: nếu khác xa tỷ lệ chung thì sự bão hòa có thể là 'lối tắt'.
    """
    y = df[LABEL_COL].to_numpy()
    base = float(y.mean())
    out = {}
    for c in cols:
        x = df[c].to_numpy(dtype="float64", na_value=np.nan)
        ok = np.isfinite(x)
        if not ok.any():
            continue
        mx = x[ok].max()
        if mx <= 0:
            continue
        m = ok & (x >= tol * mx)
        share = float(m.mean())
        if share >= min_share:
            out[c] = {"max": float(mx), "share_of_rows": round(share, 5),
                      "attack_rate_in_group": round(float(y[m].mean()), 4), "attack_rate_overall": round(base, 4)}
    return out


def icmp_contamination(df: pd.DataFrame) -> dict:
    """Kiểm tra trường ICMP có bị 'lẫn' giá trị khi luồng không phải ICMP hay không."""
    need = {"PROTOCOL", "ICMP_IPV4_TYPE", "RETRANSMITTED_IN_BYTES"}
    if not need <= set(df.columns):
        return {}
    sub = df.loc[df["PROTOCOL"] != 1, ["ICMP_IPV4_TYPE", "RETRANSMITTED_IN_BYTES"]]
    nz = sub["ICMP_IPV4_TYPE"] != 0
    eq = nz & (sub["ICMP_IPV4_TYPE"] == sub["RETRANSMITTED_IN_BYTES"])
    return {
        "non_icmp_rows": int(len(sub)),
        "icmp_type_nonzero_on_non_icmp": int(nz.sum()),
        "of_which_equal_to_retransmitted_in_bytes": int(eq.sum()),
    }


def high_corr_pairs(df: pd.DataFrame, cols: list[str], threshold: float, sample_n: int, seed: int = 0):
    s = df[cols].sample(n=min(sample_n, len(df)), random_state=seed)
    s = s.replace([np.inf, -np.inf], np.nan).fillna(s.median(numeric_only=True))
    corr = s.rank().corr().to_numpy()  # Pearson trên hạng = Spearman
    pairs = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr[i, j]
            if np.isfinite(r) and abs(r) >= threshold:
                pairs.append((cols[i], cols[j], round(float(r), 4)))
    return sorted(pairs, key=lambda t: -abs(t[2]))


def audit_dataframe(df: pd.DataFrame, cfg_audit: dict, family_info: dict | None = None) -> dict:
    feats = feature_columns(df)
    zero_thr = cfg_audit.get("sparse_zero_threshold", 0.95)
    corr_thr = cfg_audit.get("high_corr_threshold", 0.90)
    sample_n = cfg_audit.get("sample_rows", 200_000)
    auc_thr = cfg_audit.get("shortcut_auc_threshold", 0.90)
    abs_thr = float(cfg_audit.get("implausible_abs_value", 1.0e15))

    rep = {
        "overview": {
            "rows": int(len(df)),
            "columns": int(df.shape[1]),
            "memory_mb": round(float(df.memory_usage(deep=True).sum()) / 1e6, 1),
        },
        "schema": {
            "expected_features_missing": [c for c in ALL_FEATURES if c not in df.columns
                                          and c not in ("IPV4_SRC_ADDR", "IPV4_DST_ADDR")],
        },
        "labels": {
            "benign": int((df[LABEL_COL] == 0).sum()),
            "attack": int((df[LABEL_COL] == 1).sum()),
            "attack_ratio": float(df[LABEL_COL].mean()),
        },
        "attack_counts": {str(k): int(v) for k, v in df[ATTACK_COL].value_counts().items()},
    }
    if FAMILY_COL in df.columns:
        rep["family_counts"] = {str(k): int(v) for k, v in df[FAMILY_COL].value_counts().items()}
    if family_info:
        rep["family_mapping"] = family_info

    bad, by_col = nonfinite_mask(df, feats)
    rep["nonfinite"] = {"rows": int(bad.sum()), "by_col": by_col}

    neg = {}
    for c in feats:
        n = int((df[c].to_numpy(dtype="float64", na_value=np.nan) < 0).sum())
        if n:
            neg[c] = n
    rep["negative_values"] = neg

    rep["constant_cols"] = [c for c in feats if df[c].nunique(dropna=True) <= 1]
    rep["sparse_cols"] = {
        c: round(float((df[c] == 0).mean()), 4) for c in feats if (df[c] == 0).mean() >= zero_thr
    }

    # trùng lặp: trên dữ liệu thô và sau khi bỏ định danh/TTL
    raw_cols = feats + [LABEL_COL, ATTACK_COL]
    rep["duplicates_raw"] = int(df.duplicated(subset=raw_cols).sum())
    reduced = [c for c in feats if c not in HARD_DROP_COLS]
    rep["duplicates_after_dropping_identifiers_ttl"] = int(
        df.duplicated(subset=reduced + [LABEL_COL, ATTACK_COL]).sum()
    )
    rep["unique_vectors_after_dropping_identifiers_ttl"] = int(len(df) - rep["duplicates_after_dropping_identifiers_ttl"])
    rep["value_ranges"], rep["implausible_large_values"] = value_ranges(df, feats, sample_n, abs_thr)
    rep["icmp_contamination"] = icmp_contamination(df)
    rep["saturation"] = saturation_report(df, feats)
    mask, n_groups = label_conflicts(df.drop_duplicates(subset=reduced + [LABEL_COL, ATTACK_COL]), reduced)
    rep["label_conflicts"] = {"rows": int(mask.sum()), "groups": n_groups}

    auc = single_feature_auc(df, feats, sample_n)
    rep["single_feature_auc_top15"] = dict(list(auc.items())[:15])
    rep["shortcut_auc_threshold"] = auc_thr
    rep["shortcut_suspects"] = [c for c, a in auc.items() if a >= auc_thr]
    rep["high_corr_pairs"] = high_corr_pairs(df, [c for c in reduced if c not in rep["constant_cols"]],
                                             corr_thr, sample_n)[:40]
    return rep


def render_markdown(rep: dict, name: str) -> str:
    L = []
    L.append(f"# Báo cáo kiểm toán dữ liệu: {name}\n")
    o = rep["overview"]
    L.append(f"- Số dòng: **{o['rows']:,}**, số cột: {o['columns']}, bộ nhớ: {o['memory_mb']} MB")
    lb = rep["labels"]
    L.append(f"- Bình thường: {lb['benign']:,} | Tấn công: {lb['attack']:,} | Tỷ lệ tấn công: **{100*lb['attack_ratio']:.2f}%**")
    if rep["schema"]["expected_features_missing"]:
        L.append(f"- Thiếu cột so với lược đồ 43: {rep['schema']['expected_features_missing']}")
    L.append("\n## Phân bố lớp tấn công\n")
    L.append("| Lớp | Số luồng |\n|---|---|")
    for k, v in rep["attack_counts"].items():
        L.append(f"| {k} | {v:,} |")
    if "family_mapping" in rep:
        fm = rep["family_mapping"]
        L.append(f"\n- Tên tấn công không khớp mẫu họ nào (vào *Other*): {fm['unmapped_to_other']}")
        L.append(f"- Dòng có nhãn nhị phân mâu thuẫn với họ Benign: {fm['label_family_inconsistent_rows']}")
    L.append("\n## Chất lượng dữ liệu\n")
    L.append(f"- Dòng có giá trị không hữu hạn (inf/NaN): {rep['nonfinite']['rows']:,} {rep['nonfinite']['by_col']}")
    L.append(f"- Cột có giá trị âm: {rep['negative_values'] or 'không có'}")
    L.append(f"- Cột hằng (phương sai 0): {rep['constant_cols'] or 'không có'}")
    L.append(f"- Cột thưa (tỷ lệ giá trị 0 >= ngưỡng): {rep['sparse_cols'] or 'không có'}")
    L.append(f"- Dòng trùng lặp (thô): {rep['duplicates_raw']:,}")
    L.append(f"- Dòng trùng lặp sau khi bỏ định danh và TTL: {rep['duplicates_after_dropping_identifiers_ttl']:,}")
    L.append(f"- Xung đột nhãn (cùng đặc trưng, khác nhãn): {rep['label_conflicts']['rows']:,} dòng, {rep['label_conflicts']['groups']:,} nhóm")
    u = rep["unique_vectors_after_dropping_identifiers_ttl"]
    L.append(f"- Vector đặc trưng duy nhất sau khi bỏ định danh và TTL: {u:,} ({100*u/o['rows']:.1f}% số dòng)")
    L.append(f"- Giá trị |x| vượt ngưỡng vật lý (nghi giá trị đệm): {rep['implausible_large_values'] or 'không có'}")
    ic = rep["icmp_contamination"]
    if ic:
        L.append(f"- Trường ICMP trên luồng không phải ICMP: {ic['icmp_type_nonzero_on_non_icmp']:,}/{ic['non_icmp_rows']:,} dòng có "
                 f"ICMP_IPV4_TYPE khác 0, trong đó {ic['of_which_equal_to_retransmitted_in_bytes']:,} dòng bằng RETRANSMITTED_IN_BYTES")
    sat = rep.get("saturation", {})
    if sat:
        L.append("\n## Đặc trưng có nhiều dòng sát giá trị lớn nhất (bão hòa hoặc tràn số)\n")
        L.append("| Đặc trưng | max | Tỷ lệ dòng | Tỷ lệ tấn công trong nhóm | Tỷ lệ tấn công chung |\n|---|---|---|---|---|")
        for k, v in sat.items():
            L.append(f"| {k} | {v['max']:.6g} | {100*v['share_of_rows']:.2f}% | {100*v['attack_rate_in_group']:.1f}% | {100*v['attack_rate_overall']:.1f}% |")
    L.append("\n## Đặc trưng nghi là lối tắt (AUC đơn đặc trưng)\n")
    L.append("| Đặc trưng | AUC |\n|---|---|")
    for k, v in rep["single_feature_auc_top15"].items():
        L.append(f"| {k} | {v:.4f} |")
    L.append(f"\nNghi lối tắt (AUC >= {rep['shortcut_auc_threshold']}): {rep['shortcut_suspects'] or 'không có'}")
    L.append("\n## Khoảng giá trị từng đặc trưng\n")
    L.append("| Đặc trưng | min | trung vị | p99 | max |\n|---|---|---|---|---|")
    for k, v in rep["value_ranges"].items():
        L.append(f"| {k} | {v['min']:.6g} | {v['p50']:.6g} | {v['p99']:.6g} | {v['max']:.6g} |")
    L.append("\n## Cặp đặc trưng tương quan cao (Spearman)\n")
    L.append("| Đặc trưng A | Đặc trưng B | rho |\n|---|---|---|")
    for a, b, r in rep["high_corr_pairs"]:
        L.append(f"| {a} | {b} | {r} |")
    return "\n".join(L) + "\n"
