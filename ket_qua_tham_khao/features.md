# Báo cáo chọn đặc trưng (Bước 2)

Bộ dữ liệu dùng chung: unsw, cse. **25 đặc trưng được giữ**, chia thành các nhóm ngữ nghĩa dưới đây.

## Đặc trưng được giữ

| Nhóm | Đặc trưng |
|---|---|
| volume | out_pkts, out_bytes, in_byte_frac, bytes_per_pkt_out, in_pkt_frac, bytes_per_pkt_in |
| timing_rate | duration_in_ms, dst_to_src_bytes_per_s, dst_to_src_throughput, src_to_dst_bytes_per_s, src_to_dst_throughput |
| packet_size | longest_pkt, frac_pkts_le128, frac_pkts_1024_1514, frac_pkts_256_512, frac_pkts_512_1024, shortest_pkt, frac_pkts_128_256 |
| tcp_behavior | client_psh, server_psh, server_fin, server_syn, tcp_win_in |
| retransmission | retrans_in_pkts |
| protocol | is_tcp |

## Loại theo quy tắc cứng (không tạo đặc trưng)

| Cột gốc | Lý do |
|---|---|
| ICMP_TYPE | Trường ICMP mang giá trị lạ trên luồng không phải ICMP (kiểm toán: 70,8% luồng UNSW) |
| ICMP_IPV4_TYPE | Như ICMP_TYPE; PROTOCOL đã cho biết luồng ICMP |
| FTP_COMMAND_RET_CODE | Cột hằng ở CSE, chỉ có giá trị ở vài luồng FTP |
| DNS_QUERY_ID | Mã giao dịch DNS gần như ngẫu nhiên |
| DNS_QUERY_TYPE | Giá trị lạ (max 55937) và chỉ có ở luồng DNS |
| DNS_TTL_ANSWER | Giá trị tràn số (max 4,28e9 ở UNSW) và chỉ có ở luồng DNS |
| L7_PROTO | Mã danh mục phụ thuộc testbed; để dành cho thí nghiệm cắt bỏ |
| TCP_FLAGS | Là OR của CLIENT_TCP_FLAGS và SERVER_TCP_FLAGS (đã giải mã riêng) |
| RETRANSMITTED_IN_BYTES | Tương quan 0,999 với bản PKTS |
| RETRANSMITTED_OUT_BYTES | Tương quan 0,9997 với bản PKTS |

## Loại theo dữ liệu (chỉ dùng tập train)

| Đặc trưng ứng viên | Lý do |
|---|---|
| min_ip_pkt_len | unsw: lối tắt (AUC >= 0.9) |
| client_rst | unsw: thưa (>= 95% giá trị 0) |
| client_urg | unsw: hằng, thưa (>= 95% giá trị 0); cse: thưa (>= 95% giá trị 0) |
| server_rst | unsw: thưa (>= 95% giá trị 0) |
| server_urg | unsw: hằng, thưa (>= 95% giá trị 0); cse: hằng, thưa (>= 95% giá trị 0) |
| retrans_out_pkts | cse: thưa (>= 95% giá trị 0) |
| is_icmp | unsw: thưa (>= 95% giá trị 0); cse: thưa (>= 95% giá trị 0) |
| is_other_proto | unsw: thưa (>= 95% giá trị 0); cse: thưa (>= 95% giá trị 0) |
| in_bytes | trùng lặp với out_pkts (|Spearman| >= 0.9) |
| in_pkts | trùng lặp với out_pkts (|Spearman| >= 0.9) |
| flow_duration_ms | trùng lặp với duration_in_ms (|Spearman| >= 0.9) |
| duration_out_ms | trùng lặp với duration_in_ms (|Spearman| >= 0.9) |
| max_ip_pkt_len | trùng lặp với longest_pkt (|Spearman| >= 0.9) |
| client_fin | trùng lặp với server_fin (|Spearman| >= 0.9) |
| client_syn | trùng lặp với server_syn (|Spearman| >= 0.9) |
| client_ack | trùng lặp với tcp_win_in (|Spearman| >= 0.9) |
| server_ack | trùng lặp với tcp_win_in (|Spearman| >= 0.9) |
| tcp_win_out | trùng lặp với tcp_win_in (|Spearman| >= 0.9) |
| is_udp | trùng lặp với is_tcp (|Spearman| >= 0.9) |

## Sàng từng dataset (AUC có trọng số, tỷ lệ giá trị 0 có trọng số)

### unsw

| Đặc trưng | AUC (có trọng số) | AUC (không trọng số) | Tỷ lệ 0 |
|---|---|---|---|
| min_ip_pkt_len | 0.9884 | 0.9358 | 2.4% |
| in_pkt_frac | 0.8803 | 0.6953 | 0.0% |
| server_psh | 0.8664 | 0.6256 | 29.7% |
| in_byte_frac | 0.8106 | 0.5854 | 0.0% |
| bytes_per_pkt_out | 0.8101 | 0.5595 | 2.4% |
| dst_to_src_throughput | 0.7958 | 0.6033 | 2.4% |
| dst_to_src_bytes_per_s | 0.7863 | 0.6811 | 2.4% |
| out_bytes | 0.7839 | 0.6748 | 2.4% |
| out_pkts | 0.7833 | 0.6664 | 2.4% |
| client_psh | 0.7551 | 0.6199 | 28.3% |
| retrans_out_pkts | 0.7466 | 0.6683 | 28.0% |
| in_pkts | 0.7050 | 0.7083 | 0.0% |
| retrans_in_pkts | 0.6884 | 0.7271 | 29.2% |
| src_to_dst_throughput | 0.6873 | 0.6753 | 0.0% |
| src_to_dst_bytes_per_s | 0.6867 | 0.7290 | 0.0% |
| in_bytes | 0.6658 | 0.7170 | 0.0% |
| shortest_pkt | 0.6245 | 0.8891 | 0.0% |
| longest_pkt | 0.6233 | 0.7621 | 0.0% |
| max_ip_pkt_len | 0.6233 | 0.7621 | 0.0% |
| server_ack | 0.6141 | 0.7712 | 22.0% |
| is_tcp | 0.6060 | 0.7690 | 21.8% |
| client_ack | 0.6056 | 0.7751 | 21.9% |
| frac_pkts_1024_1514 | 0.5902 | 0.6148 | 61.4% |
| tcp_win_in | 0.5737 | 0.6941 | 24.4% |
| is_other_proto | 0.5704 | 0.5042 | 99.4% |
| frac_pkts_le128 | 0.5664 | 0.7572 | 0.7% |
| frac_pkts_256_512 | 0.5514 | 0.6134 | 69.7% |
| bytes_per_pkt_in | 0.5378 | 0.6340 | 0.0% |
| is_udp | 0.5289 | 0.7749 | 78.9% |
| tcp_win_out | 0.5247 | 0.7639 | 21.9% |
| client_syn | 0.5225 | 0.7382 | 47.1% |
| server_syn | 0.5220 | 0.7468 | 47.3% |
| server_fin | 0.5216 | 0.7734 | 45.7% |
| client_fin | 0.5213 | 0.7747 | 45.7% |
| flow_duration_ms | 0.5210 | 0.5440 | 94.3% |
| duration_in_ms | 0.5205 | 0.5387 | 94.4% |
| duration_out_ms | 0.5148 | 0.5364 | 94.5% |
| frac_pkts_128_256 | 0.5142 | 0.7121 | 77.6% |
| frac_pkts_512_1024 | 0.5136 | 0.5829 | 81.7% |
| is_icmp | 0.5067 | 0.5017 | 99.9% |
| client_rst | 0.5000 | 0.5011 | 100.0% |
| client_urg | 0.5000 | 0.5000 | 100.0% |
| server_rst | 0.5000 | 0.5000 | 100.0% |
| server_urg | 0.5000 | 0.5000 | 100.0% |

### cse

| Đặc trưng | AUC (có trọng số) | AUC (không trọng số) | Tỷ lệ 0 |
|---|---|---|---|
| duration_in_ms | 0.8901 | 0.9262 | 87.3% |
| tcp_win_in | 0.8819 | 0.8943 | 41.3% |
| server_fin | 0.8806 | 0.8688 | 74.5% |
| flow_duration_ms | 0.8782 | 0.8892 | 86.8% |
| frac_pkts_512_1024 | 0.8682 | 0.8979 | 62.5% |
| duration_out_ms | 0.8341 | 0.9073 | 88.4% |
| frac_pkts_256_512 | 0.8311 | 0.8770 | 62.9% |
| src_to_dst_throughput | 0.7846 | 0.8555 | 0.0% |
| client_fin | 0.7653 | 0.8414 | 67.7% |
| client_psh | 0.7579 | 0.8537 | 52.4% |
| server_psh | 0.7571 | 0.8533 | 52.5% |
| server_ack | 0.7480 | 0.8533 | 47.6% |
| client_ack | 0.7377 | 0.8532 | 48.8% |
| server_syn | 0.7356 | 0.8534 | 48.5% |
| frac_pkts_128_256 | 0.7296 | 0.7004 | 48.1% |
| dst_to_src_bytes_per_s | 0.7190 | 0.8452 | 7.5% |
| is_tcp | 0.7133 | 0.8484 | 41.3% |
| client_syn | 0.7132 | 0.8484 | 41.4% |
| is_udp | 0.7128 | 0.8484 | 58.8% |
| out_bytes | 0.7085 | 0.8170 | 7.5% |
| src_to_dst_bytes_per_s | 0.6987 | 0.8936 | 0.0% |
| in_pkts | 0.6885 | 0.8710 | 0.0% |
| out_pkts | 0.6885 | 0.8308 | 7.5% |
| in_bytes | 0.6863 | 0.8763 | 0.0% |
| bytes_per_pkt_out | 0.6829 | 0.7892 | 7.5% |
| longest_pkt | 0.6683 | 0.7960 | 0.0% |
| max_ip_pkt_len | 0.6683 | 0.7960 | 0.0% |
| in_byte_frac | 0.6639 | 0.5350 | 0.0% |
| dst_to_src_throughput | 0.6581 | 0.8449 | 7.5% |
| shortest_pkt | 0.6562 | 0.8116 | 0.0% |
| tcp_win_out | 0.6310 | 0.8106 | 48.4% |
| frac_pkts_1024_1514 | 0.6282 | 0.5416 | 74.0% |
| bytes_per_pkt_in | 0.6243 | 0.8536 | 0.0% |
| min_ip_pkt_len | 0.6193 | 0.8105 | 7.5% |
| server_rst | 0.5576 | 0.5063 | 86.3% |
| frac_pkts_le128 | 0.5485 | 0.5000 | 0.3% |
| client_rst | 0.5348 | 0.5060 | 80.4% |
| in_pkt_frac | 0.5190 | 0.7712 | 0.0% |
| retrans_out_pkts | 0.5076 | 0.5879 | 96.0% |
| retrans_in_pkts | 0.5055 | 0.6464 | 92.5% |
| is_icmp | 0.5004 | 0.5000 | 99.9% |
| client_urg | 0.5000 | 0.5000 | 100.0% |
| server_urg | 0.5000 | 0.5000 | 100.0% |
| is_other_proto | 0.5000 | 0.5000 | 100.0% |

## Cụm đặc trưng tương quan trong cùng nhóm

| Nhóm | Đại diện | Thành viên |
|---|---|---|
| volume | out_pkts | in_bytes, in_pkts, out_pkts |
| timing_rate | duration_in_ms | flow_duration_ms, duration_in_ms, duration_out_ms |
| packet_size | longest_pkt | longest_pkt, max_ip_pkt_len |
| tcp_behavior | server_fin | client_fin, server_fin |
| tcp_behavior | server_syn | client_syn, server_syn |
| tcp_behavior | tcp_win_in | client_ack, server_ack, tcp_win_in, tcp_win_out |
| protocol | is_tcp | is_tcp, is_udp |
