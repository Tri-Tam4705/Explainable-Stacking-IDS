# Explainable Stacking IDS (xsids)

Đồ án: **Ứng dụng học máy để phát hiện xâm nhập mạng dựa trên mô hình Stacking Ensemble có giải thích** (Application of Machine Learning for Network Intrusion Detection Based on Explainable Stacking Ensemble).

Bốn mô hình cây (Random Forest, Extra Trees, XGBoost, LightGBM) được ghép bằng một meta-learner tuyến tính (stacking
out-of-fold) trên hai bộ NF-UNSW-NB15-v2 và NF-CSE-CIC-IDS2018-v2. Vì meta-learner tuyến tính trên đầu ra thô, giải thích
TreeSHAP của từng mô hình gộp lại được thành giải thích chính xác của cả hệ. Lời giải thích được kiểm tra bằng tính cộng,
độ ổn định, kiểm tra tỉnh táo và phép xóa trong phân phối so với mức tối đa (Shapley theo nhóm, oracle).

Trạng thái: **mã của mọi bước (0 đến 9) đã xong và đã chạy trên dữ liệu thật**. Hướng dẫn chạy từng bước cho người mới
nhận mã: `HUONG_DAN_CHAY.md`. Lộ trình, cách đọc từng độ đo: `docs/STEPS.md`.

## Tài liệu

- `docs/Bao_cao_DACNTT.docx`: báo cáo cuối kỳ Dự án công nghệ thông tin (5 chương, theo khuôn của khoa).
- `docs/Bao_cao_Explainable_Stacking_IDS.docx`: bản viết dạng bài báo khoa học của cùng nội dung.
- `HUONG_DAN_CHAY.md`: hướng dẫn cài đặt và chạy từng bước. `docs/STEPS.md`: lộ trình và cách đọc từng độ đo.

## Cấu trúc

```
configs/
  config.yaml      dữ liệu, làm sạch, chia tập, ngưỡng kiểm toán
  features.yaml    quy tắc chọn đặc trưng (bước 2)
  models.yaml      tham số mô hình, meta-learner, SHAP, mẫu con cho các phép kiểm tra lời giải thích (explain)
data/raw/          CSV gốc; trên GitHub là bản nén .csv.gz (Git LFS)   data/processed/<ds>/   parquet train/val/test, đặc trưng, mẫu giải thích
models/<ds>/       rf, et, xgb, lgbm, stack (.joblib)
reports/           JSON/Markdown từng bước, summary.txt, figures/
scripts/           00 đến 27, gọi qua main.py
src/xsids/
  schema.py, config.py, utils.py, audit.py
  data/            loader, labels, cleaning, split, pipeline, peek, synthetic
  features/        engineer (tạo đặc trưng), select (sàng, gộp tương quan), build
  models/          zoo (4 mô hình), stacking (OOF + meta), evaluate (chỉ số có trọng số), data, determinism
  explain/         shap_stack (gộp TreeSHAP, A(x)), common (lấy mẫu, độ đo so sánh), group_shapley (Shapley nhóm),
                   stability (ổn định thứ hạng),
                   trust (kiểm chứng A(x) như tín hiệu cảnh báo sai), checks (so sánh lời giải thích, phép xóa trong phân phối)
  report/          console, summaries (bảng chữ), plots (biểu đồ)
tests/             116 test (pytest)
main.py            điểm vào: python main.py <lệnh>
```

## Cài đặt (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py check
python -m pytest tests
```

Nếu PowerShell chặn `Activate.ps1`, chạy `Set-ExecutionPolicy -Scope Process Bypass` rồi bật lại `.venv`.
`requirements.txt` ghi phiên bản tối thiểu. `requirements-lock.txt` ghi đúng phiên bản đã chạy ra số liệu trong báo cáo
(Python 3.12.2, scikit-learn 1.9.1, XGBoost 3.4.1, LightGBM 4.7.0, shap 0.54.1).

## Dữ liệu

Hai file CSV: `NF-UNSW-NB15-v2.csv` và `NF-CSE-CIC-IDS2018-v2.csv` (nhóm tác giả bộ NetFlow chuẩn hóa công bố tại
https://staff.itee.uq.edu.au/marius/NIDS_datasets/). Đặt vào `data/raw/`, hoặc khai báo đường dẫn ở dòng `path:` của từng
dataset trong `configs/config.yaml` (dùng dấu `/`). `path` có thể trỏ thẳng vào file `.zip` hoặc `.gz` chứa một CSV.

### Lấy dữ liệu khi clone từ GitHub

Repo kèm sẵn hai file nén `data/raw/NF-UNSW-NB15-v2.csv.gz` và `data/raw/NF-CSE-CIC-IDS2018-v2.csv.gz`, lưu bằng
Git LFS (CSV gốc của CSE nặng 3,2 GB, vượt giới hạn 100 MB của Git thường và 2 GB mỗi file của LFS). Không cần giải nén:
khi `data/raw/` không có file `.csv`, chương trình tự đọc file `.csv.gz` cùng tên.

```powershell
git lfs install
git clone https://github.com/Tri-Tam4705/Explainable-Stacking-IDS.git
```

Nếu đã clone trước khi cài Git LFS, `data/raw/` chỉ chứa file con trỏ vài trăm byte; chạy `git lfs pull` trong thư mục
repo để tải dữ liệu thật. Gói GitHub miễn phí cho 10 GiB băng thông LFS mỗi tháng, tính vào tài khoản chủ repo; mỗi lần
clone đầy đủ dùng khoảng bằng dung lượng hai file nén, nên chỉ clone lại khi cần.

`NF-CSE-CIC-IDS2018-v2` có khoảng 18,9 triệu dòng. Cả hai bộ dùng toàn bộ dữ liệu (`sample_frac: 1.0`); CSE được đọc
theo khối và thu nhỏ kiểu số không mất giá trị (`compact: true`), cần khoảng 12 GB RAM.

## Chạy

```powershell
python main.py all                          # check -> step1 -> step2 -> models -> eval -> cross -> shift -> report
python main.py all --from eval              # chạy tiếp từ một bước
python main.py all --fast --datasets unsw   # chạy thử với tham số nhỏ, một dataset (bỏ qua cross và shift vì cần cả hai bộ)
```

| Lệnh | Script | Việc làm | Đầu ra chính |
|---|---|---|---|
| `check` | 00 | Kiểm tra thư viện | màn hình |
| `step1` | 01b, 02, 03 | Xem CSV, kiểm toán, làm sạch, chia train/val/test | `data/processed/<ds>/*.parquet`, `reports/audit_*`, `prepare_*` |
| `step2` | 05 | Tạo và chọn 25 đặc trưng chung | `data/processed/feature_set.json`, `reports/features.md` |
| `models` | 06, 07, 08 | 4 mô hình đơn, stacking OOF, SHAP gộp | `models/<ds>/`, `reports/baselines_*`, `stacking_*`, `explain_*` |
| `eval` | 09, 10 | Ổn định của SHAP gộp, E1/E2, ablation theo nhóm | `reports/stability_*`, `ablation_*` |
| `cross` | 11 | Đánh giá chéo bộ dữ liệu, LOFO | `reports/cross_lofo.json` |
| `shift` | 16 | Dịch chuyển phân phối (KS, PSI), ngưỡng chỉnh lại | `reports/shift.json` |
| `report` | 13 | Bảng tổng hợp và tối đa 14 biểu đồ (thêm hình nhiều seed/phép chia và A(x) khi có kết quả tùy chọn), chỉ đọc `reports/` | `reports/summary.txt`, `figures/fig_*.png` |
| `multiseed` (tùy chọn) | 17 | Huấn luyện lại bước 3+4 với seed 1 đến 7 (cùng seed 42 là 8 lần) trên cả hai bộ; seed đã có kết quả thì bỏ qua. Mỗi seed khoảng 3 phút với UNSW, 2 giờ với CSE | `reports/multiseed_<ds>.json`, `models/<ds>_seed<k>/` |
| `diagnose` (tùy chọn) | 18 | Chẩn đoán: vì sao trọng số meta-learner âm, tấn công bị bỏ sót thuộc loại nào | `reports/diagnostics_*.json` |
| `retrain` (tùy chọn) | 19 | Ablation có huấn luyện lại: bỏ từng nhóm đặc trưng (và nhóm đặc trưng nghi là đặc điểm testbed), đánh giá trong miền và chéo bộ | `reports/retrain_ablation_*.json` |
| `multisplit` (tùy chọn) | 20 | Lặp lại trên 10 phép chia train/val/test; so stacking với từng mô hình đơn bằng kiểm định t ghép cặp, t đã hiệu chỉnh Nadeau-Bengio và Wilcoxon | `reports/multisplit_*.json` |
| `explaincheck` (tùy chọn) | 22, 23 | Kiểm tra lời giải thích: SHAP path-dependent so với interventional, phép xóa trong phân phối (láng giềng lành tính; k = 1 đến 10 đặc trưng và trọn 1–2 nhóm, có khoảng tin cậy bootstrap), độ bền trước nhiễu ±1%/±5%, ổn định qua 8 seed, kiểm tra tỉnh táo bằng nhãn xáo trộn | `reports/explain_checks_*.json`, `sanity_*.json`, `figures/fig_explain_checks.png`, `fig_deletion_knn.png` |
| `groupshap` (tùy chọn) | 24 | Shapley theo nhóm đặc trưng với phép xóa trong phân phối (đủ 64 liên minh, tính chính xác): phép xóa 1 đến 3 nhóm so với oracle và kỳ vọng ngẫu nhiên, tương quan trung thực, độ ổn định khi đổi tập tham chiếu; kiểm tra chéo bằng kernel Shapley; KernelSHAP mức đặc trưng so với SHAP gộp interventional | `reports/group_shapley_*.json` |
| `attacktype` (tùy chọn) | 26 | Bước hai đa lớp gọi tên loại tấn công cho luồng bị gắn cờ (LightGBM, cấu hình chọn trên val); trần do vector trùng nhiều tên; tỷ lệ đầu cuối; nhóm SHAP lớn nhất cho từng loại | `reports/attack_type_*.json` |
| `labelaudit` (tùy chọn) | 27 | Kiểm toán nhãn: luồng trùng 25 đặc trưng với luồng mang nhãn khác (trần recall), học tự tin trên xác suất out-of-fold, chỉ số khi bỏ và khi huấn luyện lại không có vector nghi sai nhãn | `reports/label_audit_*.json` |
| `trust` (tùy chọn) | 21 | A(x) có báo trước cảnh báo sai không (so với độ tự tin z); SHAP gộp của tấn công bị bỏ sót | `reports/trust_*.json`, `figures/fig_trust.png`, `fig_missed.png` |

`multiseed`, `diagnose`, `retrain`, `multisplit`, `trust`, `explaincheck`, `groupshap`, `attacktype` và `labelaudit` không nằm trong `all` và không ghi đè mô hình hay kết quả chính. Công cụ chẩn đoán tính tái lập:
`python scripts\14_check_determinism.py --dataset unsw`.

Thử pipeline bằng dữ liệu giả lập (không phải dữ liệu thật):

```powershell
python scripts\01_make_synthetic.py --dataset unsw --n 200000
python scripts\02_audit_data.py   --dataset unsw --raw-dir data/synthetic --tag synthetic
python scripts\03_prepare_data.py --dataset unsw --raw-dir data/synthetic --processed-dir data/processed_synthetic --tag synthetic
```

## Quy tắc thiết kế (chống rò rỉ, giữ tỷ lệ tự nhiên)

1. Cột định danh (IP, cổng) và TTL bị bỏ ngay từ đầu vì là đặc trưng "lối tắt"; báo cáo kiểm toán đo AUC từng đặc trưng để có bằng chứng.
2. Giá trị không hợp lệ (inf, NaN, và |x| > `max_abs_value` kiểu giá trị đệm 1.8e308) bị loại. Ép `float32` chỉ khi không tràn.
3. Các dòng có **vector đặc trưng giống hệt** được gộp thành một dòng đại diện, để một vector không nằm ở cả train và test.
   Cột `n_orig` là số dòng gốc của vector đó (trọng số), cột `n_minority` là số dòng trong nhóm có nhãn trái với nhãn đại diện.
   `conflict_policy: majority` giữ nhãn đa số (hòa thì chọn "tấn công"); `drop` bỏ hẳn các nhóm xung đột.
4. **Các chỉ số phụ thuộc tỷ lệ tấn công (DR, FAR, precision cảnh báo...) được tính có trọng số `n_orig`**; nếu tính không trọng số,
   tỷ lệ tấn công bị đẩy lên vì lưu lượng bình thường lặp lại nhiều hơn.
5. Chia tập **trước** mọi bước học từ dữ liệu. Chọn đặc trưng chỉ fit trên train; ngưỡng quyết định chọn trên val; test chỉ dùng để báo cáo.
6. `sample_frac` là lấy mẫu đều nên giữ nguyên tỷ lệ tấn công tự nhiên.
7. Giá trị thay thế khi xóa đặc trưng (trung vị lành tính, láng giềng lành tính: lấy từ train) và khi cắt đặc trưng
   (ablation, LOFO: median của val) không lấy từ test.

Lưu ý YAML: viết số mũ là `1.0e+15` (có dấu `+`); `1.0e15` bị PyYAML đọc thành chuỗi.

## Điều cần biết khi dùng kết quả

- **Độ ổn định của SHAP gộp** (`stability_*.json`) là độ ổn định của thứ hạng toàn cục khi bootstrap tập luồng, kèm phép kiểm tra
  tính lại từng luồng. Chi tiết: `docs/STEPS.md`.
- **Ablation và LOFO** thay cột bằng median của val (mặc định) rồi chọn lại ngưỡng, không huấn luyện lại. `--fill zero` tái hiện cách cũ
  (thay bằng 0), cho mức giảm lớn hơn nhiều vì 0 nằm ngoài phân phối. Cả hai đều đo độ nhạy của mô hình đã huấn luyện, không phải
  giá trị của đặc trưng nếu huấn luyện lại không có nó.
- **Ngưỡng quyết định** là điểm giữa của điểm số cho F1 tốt nhất trên val và điểm số liền dưới nó (`best_threshold`). Trên val hai
  cách cho cùng dự đoán, nhưng điểm giữa không nhạy với sai số dấu phẩy động khi nhiều luồng test có điểm đúng bằng ngưỡng
  (đã gặp ở Extra Trees trên UNSW: DR đổi 0,9851 ↔ 0,9756 giữa hai lần chạy khi ngưỡng lệch ở chữ số cuối).
- **Phép xóa đặc trưng**: thay bằng trung vị lành tính tạo vector ngoài phân phối (trên UNSW, xóa 5 đặc trưng làm 100% luồng mất cảnh báo, nhưng khi thay bằng giá trị của luồng lành tính gần nhất chỉ còn 9% đến 18%). Xem `explain_checks_*.json` và `fig_deletion_knn.png`.
- **Thứ tự nhóm đặc trưng** trên CSE phụ thuộc cách tính (path-dependent, interventional, có trọng số theo dòng cảnh báo); bốn nhóm chính vượt xa hai nhóm còn lại ở mọi cách tính.
- **Nhiều phép chia**: tập train của các phép chia chồng lên nhau nên các lần lặp không độc lập; kết luận dựa trên kiểm định t đã
  hiệu chỉnh (Nadeau và Bengio, 2003). `fig_repeat_runs.png` vẽ từng lần lặp của 4 mô hình đơn và stacking.
- **Đánh giá chéo bộ dữ liệu** gần như thất bại (DR chéo gần 0, ROC-AUC dưới 0,5). `reports/shift.json` cho thấy phần lớn đặc trưng lệch
  phân phối ngay ở luồng lành tính và chỉnh lại ngưỡng không cứu được.
- **Số liệu không trùng từng chữ số giữa các lần huấn luyện lại** (đã gặp ở UNSW). Đừng trộn kết quả của hai lần huấn luyện khác nhau trong
  cùng `reports/`: sau khi chạy lại `models`, chạy lại `eval`, `cross`, `shift`, `report` và các lệnh tùy chọn đã dùng.
- `ket_qua_tham_khao/` là bản sao `reports/` của lần chạy dùng trong báo cáo, chỉ để đối chiếu; đọc `DOC_TRUOC_KHI_DUNG.md` trong thư mục đó.
