"""Bước 2: dựng bộ đặc trưng chung cho nhiều dataset và lưu parquet."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from xsids.config import project_root
from xsids.features.engineer import engineer
from xsids.features.select import build_feature_set
from xsids.schema import ATTACK_COL, FAMILY_COL, LABEL_COL, MINORITY_COL, WEIGHT_COL
from xsids.utils import ensure_dir, get_logger, write_json

log = get_logger()
META_COLS = [LABEL_COL, ATTACK_COL, FAMILY_COL, WEIGHT_COL, MINORITY_COL]
SPLITS = ["train", "val", "test"]


def load_features_cfg(path=None) -> dict:
    path = Path(path) if path else project_root() / "configs" / "features.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_features(proc_dir, datasets: list[str], feat_cfg: dict, extra: tuple[str, ...] = ()) -> dict:
    proc_dir = Path(proc_dir)
    per_ds, groups = {}, {}
    for d in datasets:
        tr = pd.read_parquet(proc_dir / d / "train.parquet")
        X, groups = engineer(tr, extra)
        per_ds[d] = (X, tr[LABEL_COL].to_numpy(), tr[WEIGHT_COL].to_numpy(dtype="float64"))
        log.info("%s: %d vector train, %d đặc trưng ứng viên", d, len(tr), X.shape[1])

    fs = build_feature_set(per_ds, groups, feat_cfg["screening"])
    fs["fixed_drop"] = feat_cfg["fixed_drop"]
    fs["datasets"] = datasets
    fs["extra_included"] = list(extra)
    log.info("Bộ đặc trưng chung: %d đặc trưng (%s)", len(fs["features"]),
             {g: sum(1 for v in fs["groups"].values() if v == g) for g in dict.fromkeys(fs["groups"].values())})

    for d in datasets:
        out_dir = ensure_dir(proc_dir / d / "features")
        for sp in SPLITS:
            df = pd.read_parquet(proc_dir / d / f"{sp}.parquet")
            X, _ = engineer(df, extra)
            out = pd.concat([X[fs["features"]].reset_index(drop=True),
                             df[META_COLS].reset_index(drop=True)], axis=1)
            out.to_parquet(out_dir / f"{sp}.parquet", index=False)
    write_json(fs, proc_dir / "feature_set.json")
    return fs


def render_report(fs: dict) -> str:
    L = ["# Báo cáo chọn đặc trưng (Bước 2)\n"]
    L.append(f"Bộ dữ liệu dùng chung: {', '.join(fs['datasets'])}. "
             f"**{len(fs['features'])} đặc trưng được giữ**, chia thành các nhóm ngữ nghĩa dưới đây.\n")
    L.append("## Đặc trưng được giữ\n")
    L.append("| Nhóm | Đặc trưng |\n|---|---|")
    by_g: dict[str, list[str]] = {}
    for f, g in fs["groups"].items():
        by_g.setdefault(g, []).append(f)
    for g, fl in by_g.items():
        L.append(f"| {g} | {', '.join(fl)} |")
    L.append("\n## Loại theo quy tắc cứng (không tạo đặc trưng)\n")
    L.append("| Cột gốc | Lý do |\n|---|---|")
    for c, why in fs["fixed_drop"].items():
        L.append(f"| {c} | {why} |")
    L.append("\n## Loại theo dữ liệu (chỉ dùng tập train)\n")
    L.append("| Đặc trưng ứng viên | Lý do |\n|---|---|")
    for c, why in fs["dropped"].items():
        L.append(f"| {c} | {'; '.join(why)} |")
    L.append("\n## Sàng từng dataset (AUC có trọng số, tỷ lệ giá trị 0 có trọng số)\n")
    for d, tab in fs["screening"].items():
        L.append(f"### {d}\n")
        L.append("| Đặc trưng | AUC (có trọng số) | AUC (không trọng số) | Tỷ lệ 0 |\n|---|---|---|---|")
        for c, v in sorted(tab.items(), key=lambda kv: -kv[1]["auc_w"]):
            L.append(f"| {c} | {v['auc_w']:.4f} | {v['auc_unw']:.4f} | {100*v['zero_frac_w']:.1f}% |")
        L.append("")
    L.append("## Cụm đặc trưng tương quan trong cùng nhóm\n")
    L.append("| Nhóm | Đại diện | Thành viên |\n|---|---|---|")
    for cl in fs["clusters"]:
        if len(cl["members"]) > 1:
            L.append(f"| {cl['group']} | {cl['representative']} | {', '.join(cl['members'])} |")
    return "\n".join(L) + "\n"
