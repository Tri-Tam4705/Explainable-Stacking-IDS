# So sánh mô hình (tập test, có trọng số = tỷ lệ tấn công tự nhiên)

Tỷ lệ tấn công tự nhiên dùng để tính precision cảnh báo: 3.98%

| Mô hình | DR | FAR | Precision cảnh báo @ tỷ lệ tự nhiên | macro-F1 | PR-AUC |
|---|---|---|---|---|---|
| rf | 0.9835 | 0.00285 | 0.9346 | 0.9776 | 0.9771 |
| et | 0.9869 | 0.00303 | 0.9310 | 0.9774 | 0.9654 |
| xgb (tốt nhất trong 4 mô hình đơn) | 0.9829 | 0.00279 | 0.9359 | 0.9778 | 0.9807 |
| lgbm | 0.9804 | 0.00271 | 0.9375 | 0.9776 | 0.9799 |
| stack_explainable | 0.9881 | 0.00295 | 0.9329 | 0.9782 | 0.9797 |
