# Kết quả tham khảo của tác giả

Đây là thư mục `reports\` thu được khi tác giả chạy `python main.py all` trên máy của mình (Windows 11, khoảng 9,5 giờ).
Dùng để đối chiếu, không phải đầu vào của dự án. Khi bạn chạy lại, kết quả của bạn nằm ở `reports\`, không ghi đè thư mục này.

Các file nên xem trước:
- `summary.txt`: bảng tóm tắt dễ so sánh nhất.
- `figures\fig_*.png`: 10 biểu đồ.

**Lưu ý quan trọng về tính nhất quán.** Ba file bị ghi đè bởi một lần chạy lại riêng các bước 06 và 07 của UNSW
sau lần `all`: `baselines_unsw.json`, `stacking_unsw.json` và `stacking_unsw.md`. Vì vậy chúng không khớp với phần còn lại.
Bộ số nhất quán là `summary.txt` và các biểu đồ trong `figures\` (tạo ngay cuối lần `all`). Ví dụ DR của stacking ở UNSW
là 0,9834 trong `summary.txt` nhưng 0,9842 trong `stacking_unsw.json`. Khi trích số, hãy lấy từ `summary.txt`.

**Ba chỗ trong thư mục này đã lỗi thời so với mã hiện tại** (kết quả mới nằm ở `reports\`):
- `stability_*.json` và dòng "Kendall τ" trong `summary.txt` (0,666 và 0,695): sinh ra từ một lỗi ghép cặp luồng ở bản cũ của
  `09_stability.py` (so thứ hạng của hai luồng khác nhau). Không dùng hai con số này.
- `ablation_*.json` và phần LOFO của `cross_lofo.json`: thay cột bằng 0. Mã hiện tại mặc định thay bằng median của val;
  muốn tái hiện số ở đây thì chạy với `--fill zero`.
- `lime_*.json`: cấu hình `proba`/`quartile`. Cấu hình hiện tại là `logit`, `quartile` (UNSW) và `decile` (CSE).

Đừng chạy `python main.py report` trực tiếp trên thư mục này vì nó sẽ tạo lại bảng từ các file lẫn nhau.
