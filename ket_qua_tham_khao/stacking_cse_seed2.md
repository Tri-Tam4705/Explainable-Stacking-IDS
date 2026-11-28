# So sánh mô hình (tập test, có trọng số = tỷ lệ tấn công tự nhiên)

Tỷ lệ tấn công tự nhiên dùng để tính precision cảnh báo: 11.92%

| Mô hình | DR | FAR | Precision cảnh báo @ tỷ lệ tự nhiên | macro-F1 | PR-AUC |
|---|---|---|---|---|---|
| rf (tốt nhất trong 4 mô hình đơn) | 0.9361 | 0.00021 | 0.9983 | 0.9809 | 0.9724 |
| et | 0.9299 | 0.00003 | 0.9998 | 0.9795 | 0.9569 |
| xgb | 0.9354 | 0.00003 | 0.9997 | 0.9811 | 0.9572 |
| lgbm | 0.9352 | 0.00005 | 0.9996 | 0.9810 | 0.9719 |
| stack_explainable | 0.9354 | 0.00003 | 0.9997 | 0.9811 | 0.9574 |
