# So sánh mô hình (tập test, có trọng số = tỷ lệ tấn công tự nhiên)

Tỷ lệ tấn công tự nhiên dùng để tính precision cảnh báo: 11.94%

| Mô hình | DR | FAR | Precision cảnh báo @ tỷ lệ tự nhiên | macro-F1 | PR-AUC |
|---|---|---|---|---|---|
| rf | 0.9310 | 0.00007 | 0.9994 | 0.9798 | 0.9569 |
| et | 0.9299 | 0.00004 | 0.9997 | 0.9795 | 0.9565 |
| xgb (tốt nhất trong 4 mô hình đơn) | 0.9340 | 0.00012 | 0.9991 | 0.9806 | 0.9585 |
| lgbm | 0.9314 | 0.00015 | 0.9988 | 0.9797 | 0.9584 |
| voting | 0.9340 | 0.00008 | 0.9994 | 0.9807 | 0.9580 |
| stack_explainable | 0.9339 | 0.00008 | 0.9994 | 0.9807 | 0.9580 |
| stack_on_probabilities | 0.9340 | 0.00009 | 0.9993 | 0.9807 | 0.9581 |
