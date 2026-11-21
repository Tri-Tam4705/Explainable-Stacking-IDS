# So sánh mô hình (tập test, có trọng số = tỷ lệ tấn công tự nhiên)

Tỷ lệ tấn công tự nhiên dùng để tính precision cảnh báo: 11.94%

| Mô hình | DR | FAR | Precision cảnh báo @ tỷ lệ tự nhiên | macro-F1 | PR-AUC |
|---|---|---|---|---|---|
| rf | 0.9337 | 0.00013 | 0.9990 | 0.9805 | 0.9565 |
| et | 0.9326 | 0.00006 | 0.9995 | 0.9803 | 0.9567 |
| xgb | 0.9326 | 0.00004 | 0.9997 | 0.9804 | 0.9579 |
| lgbm (tốt nhất trong 4 mô hình đơn) | 0.9335 | 0.00011 | 0.9991 | 0.9805 | 0.9583 |
| voting | 0.9340 | 0.00009 | 0.9993 | 0.9807 | 0.9581 |
| stack_explainable | 0.9339 | 0.00006 | 0.9995 | 0.9807 | 0.9578 |
| stack_on_probabilities | 0.9340 | 0.00009 | 0.9993 | 0.9807 | 0.9581 |
