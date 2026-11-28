# So sánh mô hình (tập test, có trọng số = tỷ lệ tấn công tự nhiên)

Tỷ lệ tấn công tự nhiên dùng để tính precision cảnh báo: 11.92%

| Mô hình | DR | FAR | Precision cảnh báo @ tỷ lệ tự nhiên | macro-F1 | PR-AUC |
|---|---|---|---|---|---|
| rf | 0.9348 | 0.00004 | 0.9997 | 0.9809 | 0.9712 |
| et | 0.9272 | 0.00002 | 0.9998 | 0.9787 | 0.9575 |
| xgb | 0.9366 | 0.00006 | 0.9996 | 0.9815 | 0.9567 |
| lgbm (tốt nhất trong 4 mô hình đơn) | 0.9356 | 0.00020 | 0.9984 | 0.9809 | 0.9726 |
| stack_explainable | 0.9362 | 0.00006 | 0.9995 | 0.9813 | 0.9571 |
