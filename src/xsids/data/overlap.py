"""Mã băm vector đặc trưng và mức trần do các vector trùng nhau mang nhãn khác nhau.

Dùng cho các kiểm tra giới hạn dữ liệu (scripts/26-27):
  - vector_hash: mã băm của vector đặc trưng, không phụ thuộc kiểu lưu (int8, float32, ...).
  - overlap_ceiling: khi nhiều luồng có CÙNG vector đặc trưng nhưng khác nhãn, không mô hình nào chỉ nhìn vector đó phân
    biệt được chúng. Dự đoán tốt nhất cho mỗi vector là nhãn có khối lượng lớn nhất; tỷ lệ khối lượng đúng của dự đoán
    này là mức trần của mọi bộ phân loại dùng cùng các đặc trưng (tính trên chính dữ liệu, nên là cận trên).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def vector_hash(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Mã băm uint64 của từng hàng trên các cột cols; ép float64 để cùng giá trị cho cùng mã băm dù kiểu lưu khác nhau."""
    return pd.util.hash_pandas_object(df[cols].astype("float64"), index=False).to_numpy()


def overlap_ceiling(h: np.ndarray, label: np.ndarray, weight: np.ndarray | None = None) -> dict:
    """Mức trần do trùng vector: với mỗi vector, nhãn có khối lượng lớn nhất được coi là dự đoán tốt nhất.

    Trả về:
      accuracy_ceiling: tỷ lệ khối lượng được dự đoán đúng bởi quy tắc trên (cận trên của độ chính xác).
      recall_ceiling[c]: tỷ lệ khối lượng của nhãn c rơi vào vector mà c là nhãn lớn nhất (cận trên của recall lớp c).
      shared_share[c]: tỷ lệ khối lượng của nhãn c nằm ở vector cũng xuất hiện với nhãn khác.
    Hòa khối lượng: chọn nhãn xuất hiện trước theo thứ tự sắp xếp, nên cận trên có thể lệch nhẹ ở các vector hòa.
    """
    w = np.ones(len(h)) if weight is None else np.asarray(weight, dtype="float64")
    t = pd.DataFrame({"h": h, "c": pd.Series(label).astype(str).to_numpy(), "w": w})
    g = t.groupby(["h", "c"], sort=False)["w"].sum().reset_index()
    n_lab = g.groupby("h", sort=False)["c"].transform("size")
    best = g.sort_values(["h", "w", "c"], ascending=[True, False, True], kind="stable").drop_duplicates("h")
    best_of = pd.Series(best["c"].to_numpy(), index=best["h"].to_numpy())
    g["is_best"] = g["c"].to_numpy() == best_of.reindex(g["h"].to_numpy()).to_numpy()
    tot = g.groupby("c")["w"].sum()
    rec = g[g["is_best"]].groupby("c")["w"].sum().reindex(tot.index, fill_value=0.0)
    shared = g[n_lab.to_numpy() > 1].groupby("c")["w"].sum().reindex(tot.index, fill_value=0.0)
    return {"accuracy_ceiling": float(g.loc[g["is_best"], "w"].sum() / g["w"].sum()),
            "recall_ceiling": {str(c): float(rec[c] / tot[c]) for c in tot.index},
            "shared_share": {str(c): float(shared[c] / tot[c]) for c in tot.index},
            "n_vectors": int(best_of.size), "n_weighted": float(g["w"].sum())}
