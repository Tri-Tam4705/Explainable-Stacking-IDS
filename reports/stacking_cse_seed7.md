# So sánh mô hình (tập test, có trọng số = tỷ lệ tấn công tự nhiên)

Tỷ lệ tấn công tự nhiên dùng để tính precision cảnh báo: 11.94%

| Mô hình | DR | FAR | Precision cảnh báo @ tỷ lệ tự nhiên | macro-F1 | PR-AUC |
|---|---|---|---|---|---|
| rf | 0.9336 | 0.00008 | 0.9994 | 0.9806 | 0.9567 |
| et | 0.9332 | 0.00005 | 0.9996 | 0.9805 | 0.9565 |
| xgb | 0.9334 | 0.00016 | 0.9987 | 0.9803 | 0.9575 |
| lgbm (tốt nhất trong 4 mô hình đơn) | 0.9337 | 0.00007 | 0.9994 | 0.9806 | 0.9587 |
| voting | 0.9337 | 0.00004 | 0.9997 | 0.9807 | 0.9580 |
| stack_explainable | 0.9335 | 0.00005 | 0.9996 | 0.9806 | 0.9577 |
| stack_on_probabilities | 0.9340 | 0.00012 | 0.9990 | 0.9806 | 0.9579 |
