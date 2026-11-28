# Kết quả tham khảo của tác giả

Đây là bản sao thư mục `reports\` của lần chạy dùng trong báo cáo đồ án (`docs\Bao_cao_DACNTT.docx`):
toàn bộ hai bộ dữ liệu (NF-UNSW-NB15-v2: 2,39 triệu luồng; NF-CSE-CIC-IDS2018-v2: 18,9 triệu luồng, không lấy mẫu),
8 seed huấn luyện và 10 phép chia train/val/test trên cả hai bộ. Máy: Windows 11, i5-12450H, 32 GB RAM.

Dùng để đối chiếu, không phải đầu vào của dự án. Khi bạn chạy lại, kết quả của bạn nằm ở `reports\`, không ghi đè thư mục này.

Các file nên xem trước:
- `summary.txt`: bảng tóm tắt mọi kết quả (tạo bằng `python main.py report`).
- `figures\fig_*.png`: 10 biểu đồ dùng trong báo cáo.

Số liệu UNSW có thể lệch ở chữ số thập phân cuối khi huấn luyện lại (xem `HUONG_DAN_CHAY.md`, mục 8).
