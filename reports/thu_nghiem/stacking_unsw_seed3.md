# So sánh mô hình (tập test, có trọng số = tỷ lệ tấn công tự nhiên)

Tỷ lệ tấn công tự nhiên dùng để tính precision cảnh báo: 3.98%

| Mô hình | DR | FAR | Precision cảnh báo @ tỷ lệ tự nhiên | macro-F1 | PR-AUC |
|---|---|---|---|---|---|
| rf | 0.9850 | 0.00283 | 0.9352 | 0.9781 | 0.9788 |
| et | 0.9873 | 0.00317 | 0.9280 | 0.9766 | 0.9642 |
| xgb (tốt nhất trong 4 mô hình đơn) | 0.9816 | 0.00268 | 0.9382 | 0.9781 | 0.9807 |
| lgbm | 0.9840 | 0.00290 | 0.9336 | 0.9774 | 0.9782 |
| voting | 0.9869 | 0.00291 | 0.9335 | 0.9781 | 0.9734 |
| stack_explainable | 0.9865 | 0.00284 | 0.9350 | 0.9784 | 0.9796 |
| stack_on_probabilities | 0.9690 | 0.00231 | 0.9456 | 0.9771 | 0.9685 |
