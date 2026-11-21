# So sánh mô hình (tập test, có trọng số = tỷ lệ tấn công tự nhiên)

Tỷ lệ tấn công tự nhiên dùng để tính precision cảnh báo: 11.94%

| Mô hình | DR | FAR | Precision cảnh báo @ tỷ lệ tự nhiên | macro-F1 | PR-AUC |
|---|---|---|---|---|---|
| rf | 0.9338 | 0.00013 | 0.9990 | 0.9805 | 0.9566 |
| et | 0.9301 | 0.00008 | 0.9994 | 0.9795 | 0.9561 |
| xgb | 0.9342 | 0.00014 | 0.9989 | 0.9806 | 0.9572 |
| lgbm (tốt nhất trong 4 mô hình đơn) | 0.9339 | 0.00013 | 0.9990 | 0.9806 | 0.9580 |
| voting | 0.9345 | 0.00013 | 0.9990 | 0.9807 | 0.9576 |
| stack_explainable | 0.9341 | 0.00011 | 0.9991 | 0.9806 | 0.9574 |
| stack_on_probabilities | 0.9345 | 0.00013 | 0.9990 | 0.9807 | 0.9576 |
