# So sánh mô hình (tập test, có trọng số = tỷ lệ tấn công tự nhiên)

Tỷ lệ tấn công tự nhiên dùng để tính precision cảnh báo: 3.98%

| Mô hình | DR | FAR | Precision cảnh báo @ tỷ lệ tự nhiên | macro-F1 | PR-AUC |
|---|---|---|---|---|---|
| rf | 0.9853 | 0.00283 | 0.9352 | 0.9782 | 0.9773 |
| et | 0.9856 | 0.00295 | 0.9326 | 0.9775 | 0.9643 |
| xgb (tốt nhất trong 4 mô hình đơn) | 0.9819 | 0.00265 | 0.9389 | 0.9784 | 0.9806 |
| lgbm | 0.9835 | 0.00281 | 0.9354 | 0.9778 | 0.9800 |
| stack_explainable | 0.9869 | 0.00280 | 0.9358 | 0.9787 | 0.9796 |
