# So sánh mô hình (tập test, có trọng số = tỷ lệ tấn công tự nhiên)

Tỷ lệ tấn công tự nhiên dùng để tính precision cảnh báo: 11.92%

| Mô hình | DR | FAR | Precision cảnh báo @ tỷ lệ tự nhiên | macro-F1 | PR-AUC |
|---|---|---|---|---|---|
| rf | 0.9352 | 0.00004 | 0.9997 | 0.9811 | 0.9721 |
| et | 0.9342 | 0.00005 | 0.9996 | 0.9807 | 0.9577 |
| xgb | 0.9360 | 0.00007 | 0.9995 | 0.9812 | 0.9558 |
| lgbm (tốt nhất trong 4 mô hình đơn) | 0.9358 | 0.00027 | 0.9979 | 0.9808 | 0.9728 |
| stack_explainable | 0.9359 | 0.00004 | 0.9997 | 0.9813 | 0.9560 |
