# So sánh mô hình (tập test, có trọng số = tỷ lệ tấn công tự nhiên)

Tỷ lệ tấn công tự nhiên dùng để tính precision cảnh báo: 11.94%

| Mô hình | DR | FAR | Precision cảnh báo @ tỷ lệ tự nhiên | macro-F1 | PR-AUC |
|---|---|---|---|---|---|
| rf | 0.9340 | 0.00019 | 0.9985 | 0.9804 | 0.9563 |
| et | 0.9327 | 0.00010 | 0.9992 | 0.9802 | 0.9560 |
| xgb | 0.9318 | 0.00003 | 0.9998 | 0.9801 | 0.9579 |
| lgbm (tốt nhất trong 4 mô hình đơn) | 0.9312 | 0.00006 | 0.9995 | 0.9799 | 0.9584 |
| voting | 0.9340 | 0.00010 | 0.9992 | 0.9806 | 0.9575 |
| stack_explainable | 0.9341 | 0.00013 | 0.9990 | 0.9806 | 0.9576 |
| stack_on_probabilities | 0.9340 | 0.00010 | 0.9992 | 0.9806 | 0.9576 |
