# So sánh mô hình (tập test, có trọng số = tỷ lệ tấn công tự nhiên)

Tỷ lệ tấn công tự nhiên dùng để tính precision cảnh báo: 11.92%

| Mô hình | DR | FAR | Precision cảnh báo @ tỷ lệ tự nhiên | macro-F1 | PR-AUC |
|---|---|---|---|---|---|
| rf (tốt nhất trong 4 mô hình đơn) | 0.9351 | 0.00006 | 0.9995 | 0.9810 | 0.9720 |
| et | 0.9274 | 0.00002 | 0.9999 | 0.9788 | 0.9571 |
| xgb | 0.9355 | 0.00010 | 0.9992 | 0.9810 | 0.9565 |
| lgbm | 0.9353 | 0.00017 | 0.9987 | 0.9808 | 0.9589 |
| stack_explainable | 0.9355 | 0.00010 | 0.9992 | 0.9810 | 0.9562 |
