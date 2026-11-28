# Hướng dẫn chạy dự án Explainable Stacking IDS

Tài liệu này dành cho người nhận mã nguồn và muốn chạy lại dự án. Các lệnh viết cho Windows PowerShell,
là môi trường tác giả đã dùng (Windows 11, Python 3.12.2).

## 1. Dự án làm gì

Hệ thống phát hiện xâm nhập mạng (IDS) huấn luyện trên dữ liệu NetFlow của hai bộ UNSW-NB15 và CSE-CIC-IDS2018.
Bốn mô hình (Random Forest, Extra Trees, XGBoost, LightGBM) được ghép bằng một meta-learner (stacking).
Quyết định của cả hệ được giải thích bằng SHAP gộp (TreeSHAP của từng mô hình cộng theo trọng số meta-learner).
Lời giải thích được kiểm tra về tính cộng, độ ổn định và độ trung thực. Mô tả chi tiết có trong `README.md` và `docs/STEPS.md`.

## 2. Cần chuẩn bị

| Thứ cần | Ghi chú |
|---|---|
| Python 3.12 | Tác giả dùng 3.12.2. Phiên bản khác chưa được thử |
| Hai file dữ liệu CSV | `NF-UNSW-NB15-v2.csv` (khoảng 442 MB) và `NF-CSE-CIC-IDS2018-v2.csv` (khoảng 3,2 GB). Không kèm trong gói vì quá nặng |
| Ổ đĩa trống | Sau khi chạy toàn bộ, thư mục dự án thêm khoảng 2,3 GB (dữ liệu đã xử lý khoảng 0,5 GB, mô hình của 8 seed trên hai bộ khoảng 1,8 GB, số đo trên máy tác giả), chưa tính hai file CSV |
| RAM | Khoảng 12 GB khi chuẩn bị CSE (18,9 triệu dòng, dùng toàn bộ: `sample_frac: 1.0`, `compact: true` trong `configs/config.yaml`). Máy 32 GB chạy được toàn bộ quy trình |
| Thời gian | Đo trên máy tác giả (i5-12450H, 32 GB RAM) với toàn bộ hai bộ dữ liệu. CSE chiếm gần hết thời gian: `models` khoảng 4 giờ 20 phút (baselines 48 phút, stacking 3 giờ, giải thích 27 phút), `eval`, `cross`, `shift` thêm khoảng 1 giờ. Các lệnh tùy chọn trên CSE: `multiseed --seeds 1 2 3 4 5 6 7` khoảng 18 giờ, `retrain` khoảng 17 giờ, `multisplit` (10 phép chia) khoảng 9 giờ, `labelaudit --retrain` khoảng 1,6 giờ, `diagnose` khoảng 1,5 giờ. Máy khác có thể khác nhiều |

## 3. Cài đặt

Mở PowerShell tại thư mục dự án (thư mục có file `main.py`):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Muốn dùng đúng phiên bản thư viện đã chạy ra các số liệu trong báo cáo, thay dòng cuối bằng
`python -m pip install -r requirements-lock.txt`. `requirements.txt` chỉ ghi phiên bản tối thiểu.

Nếu PowerShell báo không cho chạy script, chạy `Set-ExecutionPolicy -Scope Process Bypass` rồi bật lại `.venv`.
Cài xong, đầu dòng lệnh có chữ `(.venv)`. Mỗi lần mở PowerShell mới, bật lại `.venv` trước khi chạy `python`.

## 4. Đặt dữ liệu

Có hai cách, chọn một:

1. Chép hai file CSV vào thư mục `data\raw\`, giữ **đúng tên** `NF-UNSW-NB15-v2.csv` và `NF-CSE-CIC-IDS2018-v2.csv`
   (đây là cách mặc định của `configs\config.yaml`). Bản nén cùng tên thêm đuôi `.gz` cũng chạy được, không cần giải nén.
2. Để file ở chỗ khác, rồi mở `configs\config.yaml`, bỏ dấu `#` ở dòng `path:` của từng dataset và sửa thành đường dẫn thật
   (dùng dấu `/`, ví dụ `D:/du_lieu/NF-UNSW-NB15-v2.csv`).

## 5. Kiểm tra trước khi chạy

```powershell
python main.py check
python -m pytest tests
```

`check` liệt kê thư viện, mỗi dòng phải là `[OK]`. `pytest` kỳ vọng 124 test đạt. Có một cảnh báo của LightGBM về định dạng
đầu ra của SHAP, vô hại.

## 6. Chạy

**Chạy đầy đủ cả hệ thống (khuyến nghị cho kết quả cuối):**

```powershell
python main.py all
```

Lệnh chạy lần lượt các bước và dừng ở bước đầu tiên bị lỗi, đồng thời in cách chạy tiếp. Ghi nhật ký ra file để đọc lại:

```powershell
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONUTF8 = "1"
python main.py all | Tee-Object -FilePath run_log.txt
```

**Chạy từng bước hoặc chạy tiếp từ giữa chừng:**

| Lệnh | Việc làm |
|---|---|
| `python main.py check` | Kiểm tra môi trường |
| `python main.py step1` | Đọc CSV, kiểm toán, làm sạch, chia train/val/test |
| `python main.py step2` | Chọn 25 đặc trưng chung |
| `python main.py models` | Huấn luyện bốn mô hình và stacking, tính SHAP gộp |
| `python main.py eval` | Độ ổn định của SHAP gộp (bootstrap tập luồng) và ablation |
| `python main.py cross` | Đánh giá chéo bộ dữ liệu và LOFO |
| `python main.py shift` | Dịch chuyển phân phối giữa hai bộ (KS, PSI), ngưỡng chỉnh lại |
| `python main.py report` | In bảng tổng hợp, ghi `reports\summary.txt`, vẽ biểu đồ |
| `python main.py all --from eval` | Chạy tiếp từ bước `eval` đến hết |
| `python main.py multiseed` | Tùy chọn, không nằm trong `all`: huấn luyện lại với seed 1 đến 7 trên cả hai bộ (seed đã có kết quả thì bỏ qua), báo cáo trung bình ± độ lệch chuẩn và kiểm định ghép cặp. Khoảng 3 phút mỗi seed với UNSW, 2 giờ với CSE |
| `python main.py diagnose` | Tùy chọn, không nằm trong `all`: chẩn đoán trọng số meta-learner âm, tấn công bị bỏ sót |
| `python main.py retrain` | Tùy chọn: bỏ từng nhóm đặc trưng rồi huấn luyện lại, đánh giá trong miền và chéo bộ (khoảng 20 phút với UNSW, 55 phút với CSE) |
| `python main.py multisplit` | Tùy chọn: lặp lại trên 10 phép chia train/val/test khác nhau, kiểm định t đã hiệu chỉnh Nadeau-Bengio và Wilcoxon cho stacking so với từng mô hình đơn |
| `python main.py explaincheck` | Tùy chọn: kiểm tra lời giải thích (hàm giá trị, phép xóa trong phân phối theo đặc trưng và theo nhóm, nhiễu, 8 seed, nhãn xáo trộn); khoảng 2 giờ cho hai bộ |
| `python main.py groupshap` | Tùy chọn: Shapley theo nhóm trong phân phối, so với mức tối đa của phép xóa; khoảng 23 phút cho hai bộ (nên chạy sau `explaincheck` để dùng lại SHAP interventional đã tính) |
| `python main.py attacktype` | Tùy chọn: gọi tên loại tấn công cho luồng bị gắn cờ; khoảng 3 phút với UNSW, 15 phút với CSE |
| `python main.py labelaudit` | Tùy chọn: kiểm toán nhãn (trùng vector khác nhãn, học tự tin) và huấn luyện lại không có vector nghi sai nhãn; khoảng 15 phút với UNSW, 35 phút với CSE |
| `python main.py trust` | Tùy chọn: kiểm tra A(x) có báo trước cảnh báo sai không, và giải thích các tấn công bị bỏ sót bằng SHAP gộp |

`cross` và `shift` so hai bộ dữ liệu với nhau nên cần cả hai; khi chạy với `--datasets unsw` (hoặc `cse`) hai bước này tự bỏ qua.

**Chạy thử nhanh để kiểm tra quy trình** (không dùng cho báo cáo):

```powershell
python main.py all --fast --n-bootstrap 3
```

Cách này dùng mô hình nhỏ và ít lần bootstrap. Nó **ghi đè mô hình và kết quả hiện có** bằng bản nhỏ hơn,
nên sau đó phải chạy lại `python main.py all` nếu muốn kết quả đầy đủ. Thời gian của cách này trên dữ liệu thật chưa được đo.

## 7. Xem kết quả

Sau khi chạy, mọi kết quả nằm trong `reports\`:

- `summary.txt`: bản tóm tắt dạng bảng, dễ so sánh nhất. Mở bằng Notepad.
- `figures\fig_*.png`: các biểu đồ (so sánh mô hình, recall theo loại tấn công, độ ổn định và phép xóa của SHAP gộp, ablation...).
- Các file `*.json` và `*.md`: số liệu chi tiết từng bước.

Chạy lại riêng bảng tóm tắt và biểu đồ bất kỳ lúc nào (chỉ đọc `reports\`, mất vài giây): `python main.py report`.

Thư mục `ket_qua_tham_khao\` là bản sao `reports\` của lần chạy dùng trong báo cáo (toàn bộ hai bộ dữ liệu, 8 seed, 10 phép chia),
để đối chiếu khi bạn chạy lại. Đọc `ket_qua_tham_khao\DOC_TRUOC_KHI_DUNG.md` trước khi dùng.

## 8. Điều cần biết khi đọc kết quả

- **Số liệu không trùng từng chữ số giữa các lần chạy.** Trên cùng dữ liệu và mã, UNSW cho kết quả dao động nhẹ
  (ví dụ trọng số Random Forest của meta-learner từ −0,33 đến −0,37, DR của stacking từ 0,9834 đến 0,9843).
  CSE dao động ít hơn. Đã xác định được một nguồn (LightGBM cho dự đoán out-of-fold không hoàn toàn giống nhau giữa các lần),
  nhưng chưa giải thích hết. Kết luận chính (stacking ngang các mô hình đơn tốt nhất; mô hình không khái quát
  hóa giữa hai bộ) không đổi qua các lần chạy.
- **Ngưỡng quyết định là điểm giữa hai điểm số liền kề trên val**, không phải chính một điểm số. Ở bản mã trước, DR của Extra Trees
  trên UNSW đổi từ 0,9851 sang 0,9756 giữa hai lần chạy (PR-AUC giống hệt) vì ngưỡng trùng đúng điểm của nhiều luồng test và
  lệch ở chữ số thập phân cuối cùng. Cách chọn hiện tại không còn nhạy với sai số này.
- **Đừng trộn kết quả của hai lần chạy khác nhau trong cùng một thư mục `reports\`.** Nếu chạy lại riêng một bước
  (ví dụ `models`) rồi chạy `report`, bảng tổng hợp sẽ lẫn số của hai lần chạy.
- **Công cụ chẩn đoán tính tái lập:** `python scripts\14_check_determinism.py --dataset unsw` chạy lại bước out-of-fold
  nhiều lần và chỉ ra mô hình nào không tất định. Nó không ghi đè mô hình hay báo cáo.
- **Ablation và LOFO** thay cột bằng median của tập val (mặc định) rồi chọn lại ngưỡng, không huấn luyện lại. `--fill zero` thay bằng 0,
  cho mức giảm lớn hơn nhiều vì 0 lệch khỏi dữ liệu thật.
- **Độ ổn định của SHAP gộp** là độ ổn định của thứ hạng toàn cục khi bootstrap tập luồng (xem `docs\STEPS.md`).
- **Mẫu con cho các phép kiểm tra lời giải thích** nằm ở `configs\models.yaml` (mục `explain`): 300 luồng phân tầng theo tên
  tấn công, lấy từ mẫu giải thích của bước `models` (271 luồng UNSW, 253 luồng CSE).

## 9. Cấu trúc thư mục

```
configs\     cấu hình (config.yaml: dữ liệu, làm sạch, chia tập; models.yaml: mô hình, giải thích)
src\xsids\   mã nguồn (data, features, models, explain, report)
scripts\     các bước chạy (00 đến 27)
tests\       test tự động
docs\        tài liệu các bước
data\        dữ liệu: raw (CSV gốc), processed (đã xử lý). Ban đầu để trống
models\      mô hình đã huấn luyện. Ban đầu để trống
reports\     kết quả. Ban đầu để trống
ket_qua_tham_khao\   bản sao kết quả dùng trong báo cáo, để đối chiếu
main.py      điểm vào: python main.py <lệnh>
```

## 10. Lỗi thường gặp

| Triệu chứng | Cách xử lý |
|---|---|
| `ModuleNotFoundError: xsids` | Chạy lệnh từ thư mục dự án (có `main.py`). Nếu vẫn lỗi, chạy `pip install -e .` |
| `step1` báo không tìm thấy CSV | Kiểm tra tên file trong `data\raw\` có đúng không, hoặc đường dẫn `path:` trong `configs\config.yaml` |
| Hết bộ nhớ khi đọc CSE | Giảm `sample_frac` của `cse` trong `configs\config.yaml` (ví dụ 0.1), chạy lại từ `step1` |
| Dòng đỏ `NativeCommandError` khi dùng `2>&1` | PowerShell hiển thị cảnh báo của thư viện như lỗi; không ảnh hưởng kết quả |
| Chữ tiếng Việt bị lỗi font trong log | Chạy hai dòng `[Console]::OutputEncoding...` và `$env:PYTHONUTF8...` ở mục 6 trước khi chạy |
| `eval` chạy lâu | Khoảng 10 phút mỗi bộ dữ liệu là bình thường (tính lại SHAP cho khoảng 1000 đến 1200 luồng). `--n-bootstrap` không ảnh hưởng đáng kể đến thời gian |
