# So sánh mô hình (tập test, có trọng số = tỷ lệ tấn công tự nhiên)

Tỷ lệ tấn công tự nhiên dùng để tính precision cảnh báo: 11.94%

| Mô hình | DR | FAR | Precision cảnh báo @ tỷ lệ tự nhiên | macro-F1 | PR-AUC |
|---|---|---|---|---|---|
| rf | 0.9337 | 0.00011 | 0.9992 | 0.9805 | 0.9567 |
| et | 0.9304 | 0.00006 | 0.9995 | 0.9796 | 0.9554 |
| xgb (tốt nhất trong 4 mô hình đơn) | 0.9343 | 0.00014 | 0.9989 | 0.9806 | 0.9580 |
| lgbm | 0.9340 | 0.00015 | 0.9988 | 0.9805 | 0.9578 |
| voting | 0.9342 | 0.00014 | 0.9989 | 0.9806 | 0.9576 |
| stack_explainable | 0.9341 | 0.00013 | 0.9990 | 0.9806 | 0.9578 |
| stack_on_probabilities | 0.9342 | 0.00014 | 0.9989 | 0.9806 | 0.9576 |
