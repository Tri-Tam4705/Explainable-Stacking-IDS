# So sánh mô hình (tập test, có trọng số = tỷ lệ tấn công tự nhiên)

Tỷ lệ tấn công tự nhiên dùng để tính precision cảnh báo: 11.92%

| Mô hình | DR | FAR | Precision cảnh báo @ tỷ lệ tự nhiên | macro-F1 | PR-AUC |
|---|---|---|---|---|---|
| rf | 0.9352 | 0.00006 | 0.9995 | 0.9810 | 0.9713 |
| et | 0.9299 | 0.00003 | 0.9997 | 0.9795 | 0.9569 |
| xgb | 0.9360 | 0.00012 | 0.9991 | 0.9812 | 0.9560 |
| lgbm (tốt nhất trong 4 mô hình đơn) | 0.9326 | 0.00007 | 0.9995 | 0.9802 | 0.9751 |
| stack_explainable | 0.9368 | 0.00013 | 0.9989 | 0.9813 | 0.9572 |
