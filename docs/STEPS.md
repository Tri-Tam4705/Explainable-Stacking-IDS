# Lộ trình thực hiện và liên hệ với báo cáo

| Bước | Nội dung | Đầu ra | Phần báo cáo |
|---|---|---|---|
| 0 | Môi trường, cấu trúc source, cấu hình | repo chạy được, test qua | 3.1 |
| 1 | Nạp, kiểm toán, làm sạch, chia tập (hai dataset) | parquet train/val/test, báo cáo kiểm toán | 3.2, 3.3 (dữ liệu, tiền xử lý), 4.1 |
| 2 | Đặc trưng: giải mã cờ TCP, tỷ lệ, nhóm tương quan, chọn đại diện (fit trên train) | ma trận X, bản đồ feature -> nhóm | 3.3 (đặc trưng) |
| 3 | Mô hình cơ sở: 4 mô hình đơn | bảng hiệu năng B1-B3 | 4.2 |
| 4 | Stacking có out-of-fold, meta-learner tuyến tính | mô hình stacking, hiệu năng | 3.4, 4.2 |
| 5 | Gộp TreeSHAP theo trọng số meta-learner, kiểm tra additivity | module giải thích, test số học | 3.5 |
| 6 | Đánh giá giải thích: ổn định, fidelity, so với E1-E3 | kết quả RQ1 | 4.3 |
| 7 | Điểm đồng thuận A(x) và ablation | kết quả RQ2 | 3.6, 4.4 |
| 8 | Đánh giá chéo bộ dữ liệu và LOFO | kết quả RQ3 và khái quát hóa | 4.5 |
| 9 | Tổng hợp bảng và hình, viết chương 3 đến 5 | báo cáo hoàn chỉnh | Chương 3-5 |

Nguyên tắc: mỗi bước xong thì có test, có số liệu ghi vào `reports/`, và báo cáo chỉ dùng số liệu đã chạy thật.


## Trạng thái code

| Bước | Script | Trạng thái |
|---|---|---|
| 0-1 | `main.py step1` | ✅ xong, đã chạy trên dữ liệu thật (UNSW + CSE) |
| 2 | `main.py step2` | ✅ xong, đã chạy trên dữ liệu thật |
| 3 | `scripts/06_baselines.py` | ✅ xong, đã chạy (UNSW + CSE) |
| 4 | `scripts/07_stacking.py` | ✅ xong, đã chạy (UNSW + CSE) |
| 5 | `scripts/08_explain.py` (gộp SHAP, kiểm tra tính cộng, điểm đồng thuận) | ✅ xong, đã chạy (UNSW + CSE) |
| 9 | `scripts/13_summary.py` (bảng tổng hợp và biểu đồ so sánh, đọc từ `reports/`) | ✅ xong — `main.py report` |
| 6 | `scripts/09_stability.py` (ổn định thứ hạng toàn cục + kiểm tra tất định + fidelity + so sánh E1/E2) | ✅ xong, đã chạy (UNSW + CSE) — `main.py eval` |
| 7 | `scripts/10_ablation.py` (ablation hiệu năng + ablation A(x), điền median) | ✅ xong, đã chạy (UNSW + CSE) — `main.py eval` |
| 8 | `scripts/11_cross_lofo.py` (cross-dataset + LOFO, điền median) | ✅ xong, đã chạy — `main.py cross` |
| 8b | `scripts/16_shift.py` (dịch chuyển phân phối, ngưỡng chỉnh lại) | ✅ xong, đã chạy — `main.py shift` |
| tùy chọn | `scripts/17_multiseed.py` (nhiều seed) | ✅ xong — `main.py multiseed` |
| tùy chọn | `scripts/18_diagnostics.py` (chẩn đoán trọng số meta-learner và tấn công bị bỏ sót) | ✅ xong — `main.py diagnose` |
| 9 | viết chương 3 đến 5 của báo cáo | phần viết, ngoài phạm vi mã nguồn |

## Bước 6: độ ổn định của SHAP gộp (đã sửa)

TreeSHAP tất định: cùng mô hình, cùng luồng thì luôn cho cùng φ. Vì vậy `09_stability.py` đo hai thứ khác nhau:

| Độ đo | Cách tính | Ý nghĩa |
|---|---|---|
| Kiểm tra tính lại | Tính lại SHAP gộp trên đúng các luồng của bước 5, so thứ hạng nhóm từng luồng với bản đã lưu (Kendall τ) | Phải bằng 1. Khác 1 nghĩa là mô hình hoặc mẫu đã đổi sau bước 08 |
| Ổn định thứ hạng toàn cục | Bootstrap tập luồng (mặc định 200 lần), so thứ hạng mean \|φ\| của 6 nhóm và của 25 đặc trưng với toàn bộ mẫu; thêm Jaccard top-5 đặc trưng | Kết luận "nhóm/đặc trưng nào quan trọng nhất" có phụ thuộc vào tập luồng được chọn không |

Bản trước của script so thứ hạng của luồng i với luồng thứ i của mẫu bootstrap, tức một luồng khác chọn ngẫu nhiên. Số τ ≈ 0,67 (UNSW)
và 0,70 (CSE) của bản đó đo mức giống nhau giữa hai luồng bất kỳ, không phải độ ổn định, và **không được dùng trong báo cáo**.
Bản đó cũng tính lại SHAP 20 lần nên bước `eval` mất nhiều giờ; bản mới tính một lần.


## Chạy toàn bộ hệ thống

`python main.py all` gọi lần lượt `check`, `step1`, `step2`, `models`, `eval`, `cross`, `shift`, `report`. Cờ `--from <bước>` cho phép
chạy tiếp từ giữa chừng, các cờ `--fast`, `--datasets`, `--n-explain`, `--n-bootstrap`, `--fill`... được chuyển cho từng bước.
Thời gian tham khảo trên một máy Windows (i5-12450H, 32 GB RAM), toàn bộ hai bộ dữ liệu: `models` của CSE khoảng 4 giờ 20 phút; `eval`, `cross`, `shift` của CSE thêm khoảng 1 giờ; UNSW nhanh hơn nhiều.
Kết quả UNSW có thể khác giữa hai lần huấn luyện lại (CSE trùng khớp); nếu cần số liệu cố định, giữ nguyên `models/` và chỉ chạy `eval`, `cross`.

## Bước 9: tổng hợp và biểu đồ

`python main.py report` đọc các file trong `reports/` (và `data/processed/` cho một số hình), in bảng so sánh,
ghi `reports/summary.txt` và vẽ `reports/figures/fig_*.png`. Phần in của các bước 3 đến 8 dùng cùng bộ định dạng
(`src/xsids/report/`), nên bảng in ở từng bước giống bảng trong bản tổng hợp.

Cách đọc nhanh:
- Mô hình: ★ đánh dấu giá trị tốt nhất của mỗi cột; các mô hình thường chênh nhau rất ít, đọc cùng cột "chênh max–min" trên hình.
- Ablation và LOFO: "cắt/loại" là thay cột bằng median của val (mặc định; `--fill zero` thay bằng 0 như bản cũ) và chọn lại ngưỡng trên val, không huấn luyện lại. Đây là độ nhạy của mô hình đã huấn luyện, không phải giá trị của đặc trưng nếu huấn luyện lại không có nó.

## Bổ sung tuần cuối (các script 16 đến 27 và thay đổi ở bước 08, 10, 11)

| Việc | Script / thay đổi | Ghi chú |
|---|---|---|
| Các bước kiểm tra dùng đúng cùng luồng | `08_explain.py` lưu cột `test_idx`; bước 22 và 24 lấy cùng mẫu con 300 luồng bằng `xsids.explain.common.explain_subsample` (271 luồng UNSW, 253 luồng CSE) | Sửa lệch 29.262 so với 29.261 luồng trên CSE |
| Ablation, LOFO | `10_ablation.py`, `11_cross_lofo.py` có `--fill median` (mặc định) hoặc `zero` | Median của val thay cho 0 để giảm lệch phân phối; kết quả cũ (0) vẫn tái hiện được bằng `--fill zero` |
| Dịch chuyển phân phối | `scripts/16_shift.py` | KS, PSI từng đặc trưng; ROC-AUC chéo bộ; DR/FAR sau khi chỉnh lại ngưỡng trên val đích |
| Nhiều seed | `scripts/17_multiseed.py --dataset <ds> --seeds 1 2 3 4 5 6 7` | Cả hai bộ, toàn bộ dữ liệu. Trung bình ± độ lệch chuẩn và chênh lệch ghép cặp Stacking − từng mô hình đơn; không ghi đè mô hình chính |
| Ngưỡng bền với điểm trùng | `best_threshold` trong `src/xsids/models/evaluate.py` | Trả về điểm giữa của điểm số tốt nhất và điểm số liền dưới; dự đoán trên val không đổi, nhưng luồng test có điểm đúng bằng ngưỡng không còn đổi nhãn khi điểm lệch 1 đơn vị ở chữ số cuối |
| Chẩn đoán | `scripts/18_diagnostics.py --dataset unsw` (`main.py diagnose`) | A: tương quan, VIF, trọng số khi bỏ từng mô hình, meta-learner không âm, bootstrap trọng số. B: tấn công bị bỏ sót theo loại, có bị cả bốn mô hình bỏ sót không, có thuộc nhóm xung đột nhãn không. Không ghi đè mô hình hay kết quả chính |
| Ablation có huấn luyện lại | `scripts/19_retrain_ablation.py --dataset unsw` (`main.py retrain`) | Bỏ từng nhóm, riêng `server_psh`, và bộ ba `server_psh` + hai đặc trưng tốc độ; huấn luyện lại cả hệ, đánh giá trên test cùng bộ và test của bộ kia. Đo giá trị của đặc trưng, khác bước 7 (đo độ nhạy của mô hình đã huấn luyện) |
| Nhiều phép chia | `scripts/20_multisplit.py --dataset unsw` (`main.py multisplit`, mặc định 10 phép chia) | Gộp train + val + test rồi chia lại 70/10/20 với seed chia khác nhau; trung bình ± độ lệch chuẩn, kiểm định t ghép cặp, t đã hiệu chỉnh Nadeau-Bengio (vì tập train các lần chồng nhau) và Wilcoxon cho Stacking − từng mô hình đơn. 25 đặc trưng giữ nguyên, không chọn lại cho từng phép chia |
| Kiểm tra lời giải thích | `scripts/22_explain_checks.py`, `23_sanity_randomization.py` (`main.py explaincheck`) | SHAP path-dependent so với interventional (cùng nền 500 luồng lành tính), phép xóa trong phân phối (theo đặc trưng và theo trọn nhóm, trên cả toàn mẫu giải thích của bước 08), độ bền trước nhiễu, ổn định qua 8 seed (cần `models/<ds>_seed<k>/`), kiểm tra tỉnh táo bằng nhãn xáo trộn, thứ hạng nhóm có trọng số theo dòng cảnh báo |
| Shapley nhóm trong phân phối | `scripts/24_group_shapley.py --dataset unsw` (`main.py groupshap`) | Đủ 2^6 liên minh nhóm, láng giềng lành tính của train (tập tham chiếu tính và tập đánh giá rời nhau); phép xóa g nhóm so với oracle và kỳ vọng ngẫu nhiên; tương quan trung thực; kiểm tra chéo bằng kernel Shapley; KernelSHAP mức đặc trưng so với SHAP gộp interventional |
| Gọi tên loại tấn công | `scripts/26_attack_type.py --dataset cse` (`main.py attacktype`) | LightGBM đa lớp trên vector tấn công của train; lưới tốc độ học 0,05, số lá 31/63, trọng số lớp chọn theo macro-F1 trên val (tốc độ học 0,1 làm mô hình đa lớp phân kỳ); precision/recall/F1 có trọng số, trần do vector trùng nhiều tên, tỷ lệ đầu cuối với hệ nhị phân, TreeSHAP đa lớp |
| Kiểm toán nhãn | `scripts/27_label_audit.py --dataset cse --retrain` (`main.py labelaudit`) | Luồng trùng 25 đặc trưng với luồng mang nhãn khác; học tự tin (Northcutt và cộng sự, 2021) trên xác suất OOF 5 fold của LightGBM; chỉ số khi bỏ vector test nghi sai và khi huấn luyện lại không có chúng |
| A(x) và tấn công bỏ sót | `scripts/21_trust_missed.py --dataset unsw` (`main.py trust`) | Lấy mẫu cảnh báo sai, cảnh báo đúng và tấn công bỏ sót trên test; ROC-AUC của A(x) khi tách cảnh báo sai (so với z, và trong từng khoảng z); SHAP gộp theo nhóm của tấn công bỏ sót so với tấn công bị bắt |
