# So sánh mô hình (tập test, có trọng số = tỷ lệ tấn công tự nhiên)

Tỷ lệ tấn công tự nhiên dùng để tính precision cảnh báo: 3.98%

| Mô hình | DR | FAR | Precision cảnh báo @ tỷ lệ tự nhiên | macro-F1 | PR-AUC |
|---|---|---|---|---|---|
| rf | 0.9841 | 0.00281 | 0.9355 | 0.9780 | 0.9785 |
| et | 0.9643 | 0.00237 | 0.9441 | 0.9754 | 0.9654 |
| xgb (tốt nhất trong 4 mô hình đơn) | 0.9839 | 0.00287 | 0.9342 | 0.9775 | 0.9803 |
| lgbm | 0.9854 | 0.00301 | 0.9313 | 0.9771 | 0.9794 |
| stack_explainable | 0.9854 | 0.00276 | 0.9367 | 0.9786 | 0.9795 |
