# So sánh mô hình (tập test, có trọng số = tỷ lệ tấn công tự nhiên)

Tỷ lệ tấn công tự nhiên dùng để tính precision cảnh báo: 11.92%

| Mô hình | DR | FAR | Precision cảnh báo @ tỷ lệ tự nhiên | macro-F1 | PR-AUC |
|---|---|---|---|---|---|
| rf (tốt nhất trong 4 mô hình đơn) | 0.9357 | 0.00017 | 0.9986 | 0.9809 | 0.9725 |
| et | 0.9276 | 0.00002 | 0.9998 | 0.9788 | 0.9571 |
| xgb | 0.9362 | 0.00007 | 0.9994 | 0.9813 | 0.9574 |
| lgbm | 0.9336 | 0.00016 | 0.9988 | 0.9803 | 0.9697 |
| stack_explainable | 0.9366 | 0.00023 | 0.9982 | 0.9811 | 0.9580 |
