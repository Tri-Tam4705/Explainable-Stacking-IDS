# So sánh mô hình (tập test, có trọng số = tỷ lệ tấn công tự nhiên)

Tỷ lệ tấn công tự nhiên dùng để tính precision cảnh báo: 3.98%

| Mô hình | DR | FAR | Precision cảnh báo @ tỷ lệ tự nhiên | macro-F1 | PR-AUC |
|---|---|---|---|---|---|
| rf | 0.9855 | 0.00289 | 0.9340 | 0.9779 | 0.9780 |
| et | 0.9851 | 0.00294 | 0.9328 | 0.9774 | 0.9648 |
| xgb (tốt nhất trong 4 mô hình đơn) | 0.9803 | 0.00259 | 0.9400 | 0.9783 | 0.9802 |
| lgbm | 0.9843 | 0.00292 | 0.9333 | 0.9774 | 0.9793 |
| voting | 0.9854 | 0.00274 | 0.9372 | 0.9788 | 0.9734 |
| stack_explainable | 0.9842 | 0.00270 | 0.9380 | 0.9787 | 0.9785 |
| stack_on_probabilities | 0.9764 | 0.00274 | 0.9365 | 0.9764 | 0.9690 |
