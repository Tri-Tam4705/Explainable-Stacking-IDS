# Lộ trình thực hiện và liên hệ với báo cáo

| Bước | Nội dung | Đầu ra | Phần báo cáo |
|---|---|---|---|
| 0 | Môi trường, cấu trúc source, cấu hình | repo chạy được, test qua | 3.1 |
| 1 | Nạp, kiểm toán, làm sạch, chia tập (hai dataset) | parquet train/val/test, báo cáo kiểm toán | 3.2, 3.3 (dữ liệu, tiền xử lý), 4.1 |
| 2 | Đặc trưng: giải mã cờ TCP, tỷ lệ, nhóm tương quan, chọn đại diện (fit trên train) | ma trận X, bản đồ feature -> nhóm | 3.3 (đặc trưng) |
| 3 | Mô hình cơ sở: 4 mô hình đơn, voting, stacking thông thường | bảng hiệu năng B1-B3 | 4.2 |
| 4 | Stacking có out-of-fold, meta-learner tuyến tính | mô hình stacking, hiệu năng | 3.4, 4.2 |
| 5 | Gộp TreeSHAP theo trọng số meta-learner, kiểm tra additivity | module giải thích, test số học | 3.5 |
| 5b | LIME hộp đen trên cả hệ stacking, đối chứng với SHAP gộp (đồng thuận, ổn định, trung thực cục bộ) | `reports/lime_*.json` (hình vẽ ở bước report) | 3.5, 4.3 |
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
| 5b | `scripts/12_lime.py` (LIME + so sánh với SHAP gộp, ổn định, kiểm tra xóa đặc trưng) | ✅ xong — `main.py lime` |
| 9 | `scripts/13_summary.py` (bảng tổng hợp và biểu đồ so sánh, đọc từ `reports/`) | ✅ xong — `main.py report` |
| 6 | `scripts/09_stability.py` (ổn định thứ hạng toàn cục + kiểm tra tất định + fidelity + so sánh E1/E2) | ✅ xong, đã chạy (UNSW + CSE) — `main.py eval` |
| 7 | `scripts/10_ablation.py` (ablation hiệu năng + ablation A(x), điền median) | ✅ xong, đã chạy (UNSW + CSE) — `main.py eval` |
| 8 | `scripts/11_cross_lofo.py` (cross-dataset + LOFO, điền median) | ✅ xong, đã chạy — `main.py cross` |
| 8b | `scripts/16_shift.py` (dịch chuyển phân phối, ngưỡng chỉnh lại) | ✅ xong, đã chạy — `main.py shift` |
| tùy chọn | `scripts/17_multiseed.py` (nhiều seed), `scripts/15_lime_sweep.py` (chọn cấu hình LIME) | ✅ xong — `main.py multiseed`, `main.py sweep` |
| tùy chọn | `scripts/18_diagnostics.py` (chẩn đoán trọng số meta-learner và đặc trưng hạng 1 của LIME) | ✅ xong — `main.py diagnose` |
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


## Bước 5b: LIME

Mục đích: có một giải thích độc lập với kiến trúc để đối chiếu với SHAP gộp. SHAP gộp dựa vào cấu trúc cây và
trọng số meta-learner; LIME chỉ gọi hàm dự đoán của cả hệ stacking (hộp đen) nên không dùng chung giả định nào.

```powershell
pip install lime                                  # hoặc pip install -r requirements.txt
python main.py lime --fast --datasets unsw        # thử nhanh (60 luồng, 1000 điểm nhiễu)
python main.py lime                               # đầy đủ, cả hai dataset
python scripts\12_lime.py --dataset unsw --n 500 --num-perturbations 8000
python scripts\12_lime.py --dataset unsw --target proba    # LIME phân loại trên xác suất thay vì logit
```

Cấu hình chính nằm ở `configs/models.yaml`: mục `lime` (chung) và `lime_per_dataset` (ghi đè cho từng bộ). Hiện tại:
target `logit` cho cả hai bộ, rời rạc hóa `quartile` cho UNSW và `decile` cho CSE. Đây là cấu hình đã dùng để tạo
`reports/lime_<ds>.json`, nên `python main.py lime` tái tạo được các file đó. Theo `reports/lime_sweep_<ds>.json`, đây là
cấu hình có R² mô hình thay thế cao nhất **trong các cấu hình có rời rạc hóa** (UNSW 0,39; CSE 0,51). Ở UNSW, cấu hình không
rời rạc hóa có R² cao hơn (0,61) nhưng kém hơn ở phép xóa đặc trưng; nó được chạy riêng làm đối chứng và lưu ở
`lime_<ds>_none.json` (`--discretizer none --tag none`). Khi viết báo cáo cần nêu cả hai.

Cách làm:
- Mẫu luồng: tái tạo đúng mẫu mà `08_explain.py` đã giải thích (kiểm tra bằng cột `z` của `explain_sample.parquet`),
  rồi lấy tập con phân tầng theo tên tấn công (`lime.n_samples`).
- Nền của LIME lấy từ tập **train** (không dùng test). Cờ 0/1 (cờ TCP, `is_tcp`) được xử lý như đặc trưng phân loại,
  đặc trưng liên tục được rời rạc hóa theo tứ phân vị.
- Đại lượng giải thích: logit z (`--target logit`, mặc định, cùng thang với SHAP) hoặc xác suất tấn công σ(z) (`--target proba`).

Các độ đo (đều ghi trong `reports/lime_<ds>.json`):
| Độ đo | Cách tính | Ý nghĩa |
|---|---|---|
| Đặc trưng quan trọng | mean \|đóng góp\| chuẩn hóa, Spearman toàn cục, top-k theo họ tấn công | hai phương pháp có chọn cùng đặc trưng không |
| Mức đồng thuận | Spearman \|φ\| vs \|w\| từng luồng, Jaccard top-k, khớp dấu trong top-k, đồng thuận mức nhóm (dùng lại `agreement_score`) | so SHAP gộp với LIME theo từng luồng |
| Độ ổn định | lặp LIME với seed khác nhau, Spearman và Jaccard giữa các lần | LIME ngẫu nhiên, TreeSHAP thì không |
| Độ trung thực cục bộ | (a) R² và sai số dự đoán của mô hình thay thế; (b) xóa k đặc trưng đóng góp mạnh nhất về giá trị lành tính điển hình, đo mức giảm xác suất, so với xóa ngẫu nhiên | giải thích có thật sự phản ánh mô hình không |

Lưu ý khi viết báo cáo:
- Dù cùng giải thích logit, hệ số LIME (khi rời rạc hóa) là hệ số của chỉ báo "cùng khoảng giá trị", khác bản chất với φ của SHAP,
  nên **chỉ so thứ hạng, dấu và hiệu ứng khi xóa**, không so giá trị tuyệt đối.
- R² của mô hình thay thế thường thấp (khoảng 0,3) trên mô hình cây có ngưỡng sắc: đó là hạn chế đã biết của LIME, cần nêu.
  Đừng chỉ dựa vào R² hay sai số dự đoán cục bộ: kernel quá hẹp làm mô hình thay thế suy biến (R² = 0, sai số = 0) mà vẫn "trông" chính xác.
  Phép xóa đặc trưng độc lập với mô hình thay thế nên đáng tin hơn.
- Độ rộng kernel: thử trên tập validation, kernel hẹp (1,0) tăng R² tự báo cáo nhưng giảm ổn định top-5 và độ khớp thứ hạng
  với SHAP, kernel 0,5 suy biến; vì vậy giữ mặc định của LIME. Có thể thử bằng `--kernel-width`.

## Chạy toàn bộ hệ thống

`python main.py all` gọi lần lượt `check`, `step1`, `step2`, `models`, `lime`, `eval`, `cross`, `shift`, `report`. Cờ `--from <bước>` cho phép
chạy tiếp từ giữa chừng, các cờ `--fast`, `--datasets`, `--n-lime`, `--lime-target`, `--n-bootstrap`, `--fill`... được chuyển cho từng bước.
Thời gian tham khảo trên một máy Windows (i5-12450H, 32 GB RAM): `models` khoảng 30 phút, `lime` khoảng 10 phút, `eval` khoảng 20 phút, `cross` khoảng 6 phút, `shift` dưới 1 phút. Cả `python main.py all` mất khoảng 1 giờ 11 phút.
Kết quả UNSW có thể khác giữa hai lần huấn luyện lại (CSE trùng khớp); nếu cần số liệu cố định, giữ nguyên `models/` và chỉ chạy `lime`, `eval`, `cross`.

## Bước 9: tổng hợp và biểu đồ

`python main.py report` đọc các file trong `reports/` (và `data/processed/<ds>/lime_sample.parquet` cho một hình), in bảng so sánh,
ghi `reports/summary.txt` và vẽ `reports/figures/fig_*.png`. Phần in của các bước 3 đến 8 dùng cùng bộ định dạng
(`src/xsids/report/`), nên bảng in ở từng bước giống bảng trong bản tổng hợp.

Cách đọc nhanh:
- Mô hình: ★ đánh dấu giá trị tốt nhất của mỗi cột; các mô hình thường chênh nhau rất ít, đọc cùng cột "chênh max–min" trên hình.
- LIME so với SHAP: cột "LIME so với SHAP" = (LIME − ngẫu nhiên)/(SHAP − ngẫu nhiên) ở phép xóa đặc trưng; 100% là bằng SHAP, 0% là như ngẫu nhiên.
- Ablation và LOFO: "cắt/loại" là thay cột bằng median của val (mặc định; `--fill zero` thay bằng 0 như bản cũ) và chọn lại ngưỡng trên val, không huấn luyện lại. Đây là độ nhạy của mô hình đã huấn luyện, không phải giá trị của đặc trưng nếu huấn luyện lại không có nó.

## Bổ sung tuần cuối (các script 15–17 và thay đổi ở bước 08, 10, 11, 12)

| Việc | Script / thay đổi | Ghi chú |
|---|---|---|
| Chọn cấu hình LIME | `scripts/15_lime_sweep.py --dataset unsw` | Thử 10 cấu hình (proba/logit, quartile/decile/none, kernel, số nhiễu) trên ~40 luồng; chọn theo R² mô hình thay thế, không dùng SHAP để chọn |
| LIME và SHAP dùng đúng cùng luồng | `08_explain.py` lưu cột `test_idx`; `12_lime.py` đọc lại (nếu file cũ thì ghép theo tên tấn công và logit z) | Sửa lệch 29.262 so với 29.261 luồng trên CSE |
| LIME không rời rạc hóa | `12_lime.py --discretizer none` | Nhiễu Gaussian quanh luồng, hệ số là độ dốc theo giá trị thật |
| Ablation, LOFO | `10_ablation.py`, `11_cross_lofo.py` có `--fill median` (mặc định) hoặc `zero` | Median của val thay cho 0 để giảm lệch phân phối; kết quả cũ (0) vẫn tái hiện được bằng `--fill zero` |
| Dịch chuyển phân phối | `scripts/16_shift.py` | KS, PSI từng đặc trưng; ROC-AUC chéo bộ; DR/FAR sau khi chỉnh lại ngưỡng trên val đích |
| Nhiều seed | `scripts/17_multiseed.py --dataset unsw --seeds 1 2 3` | Trung bình ± độ lệch chuẩn và chênh lệch ghép cặp Stacking − Voting; không ghi đè mô hình chính |
| Ngưỡng bền với điểm trùng | `best_threshold` trong `src/xsids/models/evaluate.py` | Trả về điểm giữa của điểm số tốt nhất và điểm số liền dưới; dự đoán trên val không đổi, nhưng luồng test có điểm đúng bằng ngưỡng không còn đổi nhãn khi điểm lệch 1 đơn vị ở chữ số cuối |
| Chẩn đoán | `scripts/18_diagnostics.py --dataset unsw` (`main.py diagnose`) | A: tương quan, VIF, trọng số khi bỏ từng mô hình, meta-learner không âm, bootstrap trọng số. B: độ nhạy thật của mô hình khi thay từng đặc trưng, so với thứ hạng của LIME và SHAP. C: tấn công bị bỏ sót theo loại, có bị cả bốn mô hình bỏ sót không, có thuộc nhóm xung đột nhãn không. Không ghi đè mô hình hay kết quả chính |
| Ablation có huấn luyện lại | `scripts/19_retrain_ablation.py --dataset unsw` (`main.py retrain`) | Bỏ từng nhóm, riêng `server_psh`, và bộ ba `server_psh` + hai đặc trưng tốc độ; huấn luyện lại cả hệ, đánh giá trên test cùng bộ và test của bộ kia. Đo giá trị của đặc trưng, khác bước 7 (đo độ nhạy của mô hình đã huấn luyện) |
| Nhiều phép chia | `scripts/20_multisplit.py --dataset unsw` (`main.py multisplit`, mặc định 10 phép chia) | Gộp train + val + test rồi chia lại 70/10/20 với seed chia khác nhau; trung bình ± độ lệch chuẩn, kiểm định t ghép cặp, t đã hiệu chỉnh Nadeau-Bengio (vì tập train các lần chồng nhau) và Wilcoxon cho Stacking − Voting. 25 đặc trưng giữ nguyên, không chọn lại cho từng phép chia |
| Kiểm tra lời giải thích | `scripts/22_explain_checks.py`, `23_sanity_randomization.py` (`main.py explaincheck`) | SHAP path-dependent so với interventional (cùng nền 500 luồng lành tính), phép xóa trong phân phối (theo đặc trưng và theo trọn nhóm, trên cả toàn mẫu giải thích của bước 08), độ bền trước nhiễu, ổn định qua 8 seed (cần `models/<ds>_seed<k>/`), kiểm tra tỉnh táo bằng nhãn xáo trộn, thứ hạng nhóm có trọng số theo dòng cảnh báo |
| A(x) và tấn công bỏ sót | `scripts/21_trust_missed.py --dataset unsw` (`main.py trust`) | Lấy mẫu cảnh báo sai, cảnh báo đúng và tấn công bỏ sót trên test; ROC-AUC của A(x) khi tách cảnh báo sai (so với z, và trong từng khoảng z); SHAP gộp theo nhóm của tấn công bỏ sót so với tấn công bị bắt |
