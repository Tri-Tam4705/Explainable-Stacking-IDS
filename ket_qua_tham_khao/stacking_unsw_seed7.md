# So sánh mô hình (tập test, có trọng số = tỷ lệ tấn công tự nhiên)

Tỷ lệ tấn công tự nhiên dùng để tính precision cảnh báo: 3.98%

| Mô hình | DR | FAR | Precision cảnh báo @ tỷ lệ tự nhiên | macro-F1 | PR-AUC |
|---|---|---|---|---|---|
| rf | 0.9864 | 0.00296 | 0.9324 | 0.9777 | 0.9778 |
| et | 0.9850 | 0.00294 | 0.9328 | 0.9774 | 0.9648 |
| xgb (tốt nhất trong 4 mô hình đơn) | 0.9851 | 0.00286 | 0.9346 | 0.9779 | 0.9808 |
| lgbm | 0.9839 | 0.00286 | 0.9345 | 0.9776 | 0.9794 |
| stack_explainable | 0.9851 | 0.00274 | 0.9370 | 0.9786 | 0.9796 |
