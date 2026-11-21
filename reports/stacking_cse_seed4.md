# So sánh mô hình (tập test, có trọng số = tỷ lệ tấn công tự nhiên)

Tỷ lệ tấn công tự nhiên dùng để tính precision cảnh báo: 11.94%

| Mô hình | DR | FAR | Precision cảnh báo @ tỷ lệ tự nhiên | macro-F1 | PR-AUC |
|---|---|---|---|---|---|
| rf | 0.9338 | 0.00014 | 0.9989 | 0.9805 | 0.9571 |
| et | 0.9308 | 0.00003 | 0.9998 | 0.9798 | 0.9563 |
| xgb | 0.9340 | 0.00010 | 0.9992 | 0.9806 | 0.9576 |
| lgbm (tốt nhất trong 4 mô hình đơn) | 0.9336 | 0.00008 | 0.9994 | 0.9806 | 0.9581 |
| voting | 0.9341 | 0.00013 | 0.9990 | 0.9806 | 0.9579 |
| stack_explainable | 0.9338 | 0.00010 | 0.9992 | 0.9806 | 0.9578 |
| stack_on_probabilities | 0.9341 | 0.00012 | 0.9990 | 0.9806 | 0.9580 |
