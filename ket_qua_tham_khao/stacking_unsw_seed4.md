# So sánh mô hình (tập test, có trọng số = tỷ lệ tấn công tự nhiên)

Tỷ lệ tấn công tự nhiên dùng để tính precision cảnh báo: 3.98%

| Mô hình | DR | FAR | Precision cảnh báo @ tỷ lệ tự nhiên | macro-F1 | PR-AUC |
|---|---|---|---|---|---|
| rf | 0.9848 | 0.00290 | 0.9336 | 0.9776 | 0.9785 |
| et | 0.9848 | 0.00288 | 0.9340 | 0.9777 | 0.9642 |
| xgb (tốt nhất trong 4 mô hình đơn) | 0.9810 | 0.00263 | 0.9391 | 0.9782 | 0.9806 |
| lgbm | 0.9805 | 0.00274 | 0.9367 | 0.9774 | 0.9795 |
| stack_explainable | 0.9843 | 0.00267 | 0.9384 | 0.9789 | 0.9798 |
