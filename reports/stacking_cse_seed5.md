# So sánh mô hình (tập test, có trọng số = tỷ lệ tấn công tự nhiên)

Tỷ lệ tấn công tự nhiên dùng để tính precision cảnh báo: 11.92%

| Mô hình | DR | FAR | Precision cảnh báo @ tỷ lệ tự nhiên | macro-F1 | PR-AUC |
|---|---|---|---|---|---|
| rf (tốt nhất trong 4 mô hình đơn) | 0.9351 | 0.00008 | 0.9994 | 0.9809 | 0.9719 |
| et | 0.9276 | 0.00003 | 0.9998 | 0.9788 | 0.9576 |
| xgb | 0.9350 | 0.00004 | 0.9997 | 0.9810 | 0.9565 |
| lgbm | 0.9352 | 0.00019 | 0.9985 | 0.9807 | 0.9664 |
| stack_explainable | 0.9355 | 0.00015 | 0.9988 | 0.9809 | 0.9562 |
