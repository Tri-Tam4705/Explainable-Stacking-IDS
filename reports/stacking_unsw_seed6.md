# So sánh mô hình (tập test, có trọng số = tỷ lệ tấn công tự nhiên)

Tỷ lệ tấn công tự nhiên dùng để tính precision cảnh báo: 3.98%

| Mô hình | DR | FAR | Precision cảnh báo @ tỷ lệ tự nhiên | macro-F1 | PR-AUC |
|---|---|---|---|---|---|
| rf | 0.9844 | 0.00285 | 0.9346 | 0.9778 | 0.9784 |
| et | 0.9718 | 0.00260 | 0.9394 | 0.9760 | 0.9640 |
| xgb (tốt nhất trong 4 mô hình đơn) | 0.9901 | 0.00320 | 0.9277 | 0.9772 | 0.9805 |
| lgbm | 0.9804 | 0.00280 | 0.9354 | 0.9770 | 0.9796 |
| voting | 0.9853 | 0.00279 | 0.9360 | 0.9784 | 0.9735 |
| stack_explainable | 0.9862 | 0.00283 | 0.9351 | 0.9784 | 0.9793 |
| stack_on_probabilities | 0.9860 | 0.00290 | 0.9337 | 0.9779 | 0.9685 |
