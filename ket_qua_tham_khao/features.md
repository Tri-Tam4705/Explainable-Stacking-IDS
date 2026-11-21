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
| duration_in_ms | 0.8901 | 0.9337 | 87.1% |
| tcp_win_in | 0.8854 | 0.9045 | 41.6% |
| server_fin | 0.8792 | 0.8760 | 74.1% |
| flow_duration_ms | 0.8780 | 0.8952 | 86.6% |
| frac_pkts_512_1024 | 0.8684 | 0.9082 | 62.7% |
| duration_out_ms | 0.8336 | 0.9171 | 88.2% |
| frac_pkts_256_512 | 0.8298 | 0.8865 | 63.1% |
| src_to_dst_throughput | 0.7829 | 0.8686 | 0.0% |
| client_fin | 0.7660 | 0.8497 | 67.8% |
| client_psh | 0.7558 | 0.8557 | 52.0% |
| server_psh | 0.7549 | 0.8553 | 52.1% |
| server_ack | 0.7507 | 0.8554 | 48.0% |
| client_ack | 0.7368 | 0.8550 | 48.6% |
| server_syn | 0.7359 | 0.8553 | 48.4% |
| frac_pkts_128_256 | 0.7341 | 0.7062 | 47.5% |
| dst_to_src_bytes_per_s | 0.7165 | 0.8464 | 7.6% |
| client_syn | 0.7154 | 0.8490 | 41.7% |
| is_tcp | 0.7152 | 0.8490 | 41.6% |
| is_udp | 0.7147 | 0.8489 | 58.5% |
| out_bytes | 0.7058 | 0.8202 | 7.6% |
| src_to_dst_bytes_per_s | 0.6978 | 0.8904 | 0.0% |
| out_pkts | 0.6852 | 0.8332 | 7.6% |
| in_bytes | 0.6849 | 0.8734 | 0.0% |
| in_pkts | 0.6845 | 0.8681 | 0.0% |
| bytes_per_pkt_out | 0.6788 | 0.7923 | 7.6% |
| longest_pkt | 0.6647 | 0.7952 | 0.0% |
| max_ip_pkt_len | 0.6647 | 0.7952 | 0.0% |
| in_byte_frac | 0.6622 | 0.5021 | 0.0% |
| shortest_pkt | 0.6590 | 0.8061 | 0.0% |
| dst_to_src_throughput | 0.6581 | 0.8498 | 7.6% |
| tcp_win_out | 0.6381 | 0.8055 | 48.4% |
| frac_pkts_1024_1514 | 0.6303 | 0.5508 | 73.7% |
| bytes_per_pkt_in | 0.6235 | 0.8541 | 0.0% |
| min_ip_pkt_len | 0.6208 | 0.8028 | 7.6% |
| frac_pkts_le128 | 0.5576 | 0.5031 | 0.3% |
| server_rst | 0.5521 | 0.5095 | 87.2% |
| client_rst | 0.5353 | 0.5021 | 80.4% |
| in_pkt_frac | 0.5191 | 0.7376 | 0.0% |
| retrans_out_pkts | 0.5072 | 0.5749 | 96.0% |
| retrans_in_pkts | 0.5049 | 0.6127 | 92.5% |
| is_icmp | 0.5005 | 0.5001 | 99.9% |
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
