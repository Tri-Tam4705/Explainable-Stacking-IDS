# So sánh mô hình (tập test, có trọng số = tỷ lệ tấn công tự nhiên)

Tỷ lệ tấn công tự nhiên dùng để tính precision cảnh báo: 11.94%

| Mô hình | DR | FAR | Precision cảnh báo @ tỷ lệ tự nhiên | macro-F1 | PR-AUC |
|---|---|---|---|---|---|
| rf | 0.9336 | 0.00008 | 0.9994 | 0.9806 | 0.9564 |
| et | 0.9282 | 0.00006 | 0.9995 | 0.9790 | 0.9563 |
| xgb (tốt nhất trong 4 mô hình đơn) | 0.9342 | 0.00008 | 0.9994 | 0.9807 | 0.9576 |
| lgbm | 0.9326 | 0.00010 | 0.9992 | 0.9802 | 0.9575 |
| voting | 0.9341 | 0.00013 | 0.9990 | 0.9806 | 0.9577 |
| stack_explainable | 0.9346 | 0.00031 | 0.9975 | 0.9803 | 0.9576 |
| stack_on_probabilities | 0.9341 | 0.00013 | 0.9990 | 0.9806 | 0.9577 |
