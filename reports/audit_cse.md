# Báo cáo kiểm toán dữ liệu: NF-CSE-CIC-IDS2018-v2

- Số dòng: **4,721,888**, số cột: 44, bộ nhớ: 1691.0 MB
- Bình thường: 4,156,849 | Tấn công: 565,039 | Tỷ lệ tấn công: **11.97%**

## Phân bố lớp tấn công

| Lớp | Số luồng |
|---|---|
| Benign | 4,156,849 |
| DDOS attack-HOIC | 270,535 |
| DoS attacks-Hulk | 108,143 |
| DDoS attacks-LOIC-HTTP | 76,783 |
| Bot | 35,830 |
| Infilteration | 29,206 |
| SSH-Bruteforce | 23,762 |
| DoS attacks-GoldenEye | 6,958 |
| FTP-BruteForce | 6,525 |
| DoS attacks-SlowHTTPTest | 3,576 |
| DoS attacks-Slowloris | 2,315 |
| DDOS attack-LOIC-UDP | 523 |
| Brute Force -Web | 517 |
| Brute Force -XSS | 244 |
| SQL Injection | 122 |

- Tên tấn công không khớp mẫu họ nào (vào *Other*): ['Bot', 'Brute Force -Web', 'Brute Force -XSS', 'FTP-BruteForce', 'Infilteration', 'SQL Injection', 'SSH-Bruteforce']
- Dòng có nhãn nhị phân mâu thuẫn với họ Benign: 0

## Chất lượng dữ liệu

- Dòng có giá trị không hữu hạn (inf/NaN): 0 {}
- Cột có giá trị âm: không có
- Cột hằng (phương sai 0): ['FTP_COMMAND_RET_CODE']
- Cột thưa (tỷ lệ giá trị 0 >= ngưỡng): {'RETRANSMITTED_OUT_BYTES': 0.9594, 'RETRANSMITTED_OUT_PKTS': 0.9594, 'FTP_COMMAND_RET_CODE': 1.0}
- Dòng trùng lặp (thô): 263,224
- Dòng trùng lặp sau khi bỏ định danh và TTL: 2,302,173
- Xung đột nhãn (cùng đặc trưng, khác nhãn): 482 dòng, 241 nhóm
- Vector đặc trưng duy nhất sau khi bỏ định danh và TTL: 2,419,715 (51.2% số dòng)
- Giá trị |x| vượt ngưỡng vật lý (nghi giá trị đệm): {'SRC_TO_DST_SECOND_BYTES': 72, 'DST_TO_SRC_SECOND_BYTES': 28}
- Trường ICMP trên luồng không phải ICMP: 360,871/4,716,994 dòng có ICMP_IPV4_TYPE khác 0, trong đó 125,350 dòng bằng RETRANSMITTED_IN_BYTES

## Đặc trưng có nhiều dòng sát giá trị lớn nhất (bão hòa hoặc tràn số)

| Đặc trưng | max | Tỷ lệ dòng | Tỷ lệ tấn công trong nhóm | Tỷ lệ tấn công chung |
|---|---|---|---|---|
| L4_SRC_PORT | 65535 | 0.37% | 7.7% | 12.0% |
| TCP_FLAGS | 223 | 9.36% | 15.4% | 12.0% |
| CLIENT_TCP_FLAGS | 223 | 2.71% | 0.4% | 12.0% |
| FLOW_DURATION_MILLISECONDS | 4.29497e+06 | 13.40% | 73.8% | 12.0% |
| TCP_WIN_MAX_IN | 65535 | 6.51% | 88.2% | 12.0% |
| TCP_WIN_MAX_OUT | 65535 | 0.79% | 0.8% | 12.0% |
| DNS_QUERY_TYPE | 255 | 0.27% | 0.1% | 12.0% |

## Đặc trưng nghi là lối tắt (AUC đơn đặc trưng)

| Đặc trưng | AUC |
|---|---|
| DURATION_IN | 0.8993 |
| FLOW_DURATION_MILLISECONDS | 0.8868 |
| TCP_WIN_MAX_IN | 0.8847 |
| DURATION_OUT | 0.8481 |
| L7_PROTO | 0.8240 |
| SRC_TO_DST_AVG_THROUGHPUT | 0.7995 |
| NUM_PKTS_512_TO_1024_BYTES | 0.7865 |
| CLIENT_TCP_FLAGS | 0.7448 |
| NUM_PKTS_256_TO_512_BYTES | 0.7344 |
| NUM_PKTS_128_TO_256_BYTES | 0.7312 |
| TCP_FLAGS | 0.7228 |
| DNS_QUERY_ID | 0.7147 |
| DNS_QUERY_TYPE | 0.7144 |
| PROTOCOL | 0.7126 |
| SERVER_TCP_FLAGS | 0.7122 |

Nghi lối tắt (AUC >= 0.9): không có

## Khoảng giá trị từng đặc trưng

| Đặc trưng | min | trung vị | p99 | max |
|---|---|---|---|---|
| L4_SRC_PORT | 0 | 54191 | 65307 | 65535 |
| L4_DST_PORT | 0 | 80 | 9324.08 | 65535 |
| PROTOCOL | 1 | 6 | 17 | 58 |
| L7_PROTO | 0 | 0 | 131 | 248 |
| IN_BYTES | 28 | 204.5 | 4216.03 | 6.82015e+07 |
| IN_PKTS | 1 | 4 | 36 | 202654 |
| OUT_BYTES | 0 | 320 | 34429.1 | 2.64399e+08 |
| OUT_PKTS | 0 | 3 | 47 | 176600 |
| TCP_FLAGS | 0 | 27 | 223 | 223 |
| CLIENT_TCP_FLAGS | 0 | 26 | 223 | 223 |
| SERVER_TCP_FLAGS | 0 | 22 | 94 | 223 |
| FLOW_DURATION_MILLISECONDS | 0 | 0 | 4.29496e+06 | 4.29497e+06 |
| DURATION_IN | 0 | 0 | 285.01 | 106714 |
| DURATION_OUT | 0 | 0 | 281 | 23853 |
| MIN_TTL | 0 | 101 | 128 | 255 |
| MAX_TTL | 0 | 101 | 128 | 255 |
| LONGEST_FLOW_PKT | 28 | 191 | 1500 | 65212 |
| SHORTEST_FLOW_PKT | 28 | 40 | 102 | 1500 |
| MIN_IP_PKT_LEN | 0 | 40 | 94 | 330 |
| MAX_IP_PKT_LEN | 28 | 192 | 1500 | 65212 |
| SRC_TO_DST_SECOND_BYTES | 0 | 186 | 92381 | 1.30802e+300 |
| DST_TO_SRC_SECOND_BYTES | 0 | 340 | 521095 | 6.10481e+46 |
| RETRANSMITTED_IN_BYTES | 0 | 0 | 656 | 79649 |
| RETRANSMITTED_IN_PKTS | 0 | 0 | 11 | 346 |
| RETRANSMITTED_OUT_BYTES | 0 | 0 | 1213 | 2.06324e+06 |
| RETRANSMITTED_OUT_PKTS | 0 | 0 | 2 | 1379 |
| SRC_TO_DST_AVG_THROUGHPUT | 0 | 992000 | 2.5312e+07 | 3.52003e+09 |
| DST_TO_SRC_AVG_THROUGHPUT | 0 | 1.464e+06 | 1.40601e+08 | 4.29386e+09 |
| NUM_PKTS_UP_TO_128_BYTES | 0 | 6 | 49 | 202654 |
| NUM_PKTS_128_TO_256_BYTES | 0 | 1 | 7 | 6514 |
| NUM_PKTS_256_TO_512_BYTES | 0 | 0 | 6 | 1443 |
| NUM_PKTS_512_TO_1024_BYTES | 0 | 0 | 5 | 2113 |
| NUM_PKTS_1024_TO_1514_BYTES | 0 | 0 | 21 | 176145 |
| TCP_WIN_MAX_IN | 0 | 8192 | 65535 | 65535 |
| TCP_WIN_MAX_OUT | 0 | 8192 | 64800 | 65535 |
| ICMP_TYPE | 0 | 0 | 56066.6 | 65280 |
| ICMP_IPV4_TYPE | 0 | 0 | 208 | 255 |
| DNS_QUERY_ID | 0 | 0 | 63908 | 65535 |
| DNS_QUERY_TYPE | 0 | 0 | 28 | 255 |
| DNS_TTL_ANSWER | 0 | 0 | 60 | 134473 |
| FTP_COMMAND_RET_CODE | 0 | 0 | 0 | 0 |

## Cặp đặc trưng tương quan cao (Spearman)

| Đặc trưng A | Đặc trưng B | rho |
|---|---|---|
| LONGEST_FLOW_PKT | MAX_IP_PKT_LEN | 1.0 |
| ICMP_TYPE | ICMP_IPV4_TYPE | 1.0 |
| RETRANSMITTED_OUT_BYTES | RETRANSMITTED_OUT_PKTS | 0.9997 |
| RETRANSMITTED_IN_BYTES | RETRANSMITTED_IN_PKTS | 0.9991 |
| RETRANSMITTED_IN_PKTS | ICMP_TYPE | 0.9942 |
| RETRANSMITTED_IN_PKTS | ICMP_IPV4_TYPE | 0.9942 |
| RETRANSMITTED_IN_BYTES | ICMP_TYPE | 0.9937 |
| RETRANSMITTED_IN_BYTES | ICMP_IPV4_TYPE | 0.9937 |
| OUT_BYTES | DST_TO_SRC_SECOND_BYTES | 0.9901 |
| TCP_FLAGS | CLIENT_TCP_FLAGS | 0.9831 |
| IN_BYTES | SRC_TO_DST_SECOND_BYTES | 0.9756 |
| PROTOCOL | DNS_QUERY_TYPE | 0.9621 |
| FLOW_DURATION_MILLISECONDS | DURATION_IN | 0.9607 |
| IN_PKTS | OUT_PKTS | 0.9542 |
| IN_PKTS | NUM_PKTS_UP_TO_128_BYTES | 0.9521 |
| OUT_BYTES | LONGEST_FLOW_PKT | 0.9516 |
| OUT_BYTES | MAX_IP_PKT_LEN | 0.9516 |
| PROTOCOL | DNS_QUERY_ID | 0.9447 |
| OUT_PKTS | NUM_PKTS_UP_TO_128_BYTES | 0.944 |
| LONGEST_FLOW_PKT | DST_TO_SRC_SECOND_BYTES | 0.944 |
| MAX_IP_PKT_LEN | DST_TO_SRC_SECOND_BYTES | 0.944 |
| IN_BYTES | IN_PKTS | 0.9421 |
| PROTOCOL | DNS_TTL_ANSWER | 0.9408 |
| OUT_BYTES | OUT_PKTS | 0.932 |
| DNS_QUERY_ID | DNS_QUERY_TYPE | 0.9319 |
| IN_PKTS | SRC_TO_DST_SECOND_BYTES | 0.9289 |
| OUT_PKTS | DST_TO_SRC_SECOND_BYTES | 0.9258 |
| PROTOCOL | SHORTEST_FLOW_PKT | 0.9256 |
| DNS_QUERY_TYPE | DNS_TTL_ANSWER | 0.9245 |
| FLOW_DURATION_MILLISECONDS | DURATION_OUT | 0.9241 |
| IN_BYTES | OUT_PKTS | 0.921 |
| SRC_TO_DST_SECOND_BYTES | NUM_PKTS_UP_TO_128_BYTES | 0.9147 |
| OUT_PKTS | SRC_TO_DST_SECOND_BYTES | 0.9119 |
| DURATION_IN | DURATION_OUT | 0.9116 |
| DNS_QUERY_ID | DNS_TTL_ANSWER | 0.9113 |
| TCP_FLAGS | SERVER_TCP_FLAGS | 0.9043 |
| PROTOCOL | TCP_WIN_MAX_IN | -0.9025 |
| SHORTEST_FLOW_PKT | MIN_IP_PKT_LEN | 0.9024 |
| IN_BYTES | NUM_PKTS_UP_TO_128_BYTES | 0.9003 |
