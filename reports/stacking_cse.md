# So sánh mô hình (tập test, có trọng số = tỷ lệ tấn công tự nhiên)

Tỷ lệ tấn công tự nhiên dùng để tính precision cảnh báo: 11.92%

| Mô hình | DR | FAR | Precision cảnh báo @ tỷ lệ tự nhiên | macro-F1 | PR-AUC |
|---|---|---|---|---|---|
| rf | 0.9362 | 0.00018 | 0.9985 | 0.9810 | 0.9711 |
| et | 0.9300 | 0.00002 | 0.9998 | 0.9795 | 0.9570 |
| xgb | 0.9356 | 0.00009 | 0.9993 | 0.9811 | 0.9582 |
| lgbm (tốt nhất trong 4 mô hình đơn) | 0.9370 | 0.00007 | 0.9994 | 0.9815 | 0.9734 |
| stack_explainable | 0.9360 | 0.00020 | 0.9984 | 0.9810 | 0.9586 |
