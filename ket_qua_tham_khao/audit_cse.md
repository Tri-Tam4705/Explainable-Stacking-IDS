# Báo cáo kiểm toán dữ liệu: NF-CSE-CIC-IDS2018-v2

- Số dòng: **18,893,708**, số cột: 44, bộ nhớ: 3132.7 MB
- Bình thường: 16,635,567 | Tấn công: 2,258,141 | Tỷ lệ tấn công: **11.95%**

## Phân bố lớp tấn công

| Lớp | Số luồng |
|---|---|
| Benign | 16,635,567 |
| DDOS attack-HOIC | 1,080,858 |
| DoS attacks-Hulk | 432,648 |
| DDoS attacks-LOIC-HTTP | 307,300 |
| Bot | 143,097 |
| Infilteration | 116,361 |
| SSH-Bruteforce | 94,979 |
| DoS attacks-GoldenEye | 27,723 |
| FTP-BruteForce | 25,933 |
| DoS attacks-SlowHTTPTest | 14,116 |
| DoS attacks-Slowloris | 9,512 |
| Brute Force -Web | 2,143 |
| DDOS attack-LOIC-UDP | 2,112 |
| Brute Force -XSS | 927 |
| SQL Injection | 432 |

- Tên tấn công không khớp mẫu họ nào (vào *Other*): ['Bot', 'Brute Force -Web', 'Brute Force -XSS', 'FTP-BruteForce', 'Infilteration', 'SQL Injection', 'SSH-Bruteforce']
- Dòng có nhãn nhị phân mâu thuẫn với họ Benign: 0

## Chất lượng dữ liệu

- Dòng có giá trị không hữu hạn (inf/NaN): 0 {}
- Cột có giá trị âm: không có
- Cột hằng (phương sai 0): ['FTP_COMMAND_RET_CODE']
- Cột thưa (tỷ lệ giá trị 0 >= ngưỡng): {'RETRANSMITTED_OUT_BYTES': 0.9594, 'RETRANSMITTED_OUT_PKTS': 0.9594, 'FTP_COMMAND_RET_CODE': 1.0}
- Dòng trùng lặp (thô): 1,763,436
- Dòng trùng lặp sau khi bỏ định danh và TTL: 10,041,104
- Xung đột nhãn (cùng đặc trưng, khác nhãn): 1,300 dòng, 649 nhóm
- Vector đặc trưng duy nhất sau khi bỏ định danh và TTL: 8,852,604 (46.9% số dòng)
- Giá trị |x| vượt ngưỡng vật lý (nghi giá trị đệm): {'SRC_TO_DST_SECOND_BYTES': 331, 'DST_TO_SRC_SECOND_BYTES': 126}
- Trường ICMP trên luồng không phải ICMP: 1,444,650/18,873,886 dòng có ICMP_IPV4_TYPE khác 0, trong đó 501,463 dòng bằng RETRANSMITTED_IN_BYTES

## Đặc trưng có nhiều dòng sát giá trị lớn nhất (bão hòa hoặc tràn số)

| Đặc trưng | max | Tỷ lệ dòng | Tỷ lệ tấn công trong nhóm | Tỷ lệ tấn công chung |
|---|---|---|---|---|
| L4_SRC_PORT | 65535 | 0.37% | 7.8% | 11.9% |
| TCP_FLAGS | 223 | 9.34% | 15.5% | 11.9% |
| CLIENT_TCP_FLAGS | 223 | 2.70% | 0.4% | 11.9% |
| FLOW_DURATION_MILLISECONDS | 4.29497e+06 | 13.37% | 73.8% | 11.9% |
| TCP_WIN_MAX_IN | 65535 | 6.50% | 88.3% | 11.9% |
| TCP_WIN_MAX_OUT | 65535 | 0.78% | 0.8% | 11.9% |

## Đặc trưng nghi là lối tắt (AUC đơn đặc trưng)

| Đặc trưng | AUC |
|---|---|
| DURATION_IN | 0.8993 |
| FLOW_DURATION_MILLISECONDS | 0.8871 |
| TCP_WIN_MAX_IN | 0.8843 |
| DURATION_OUT | 0.8480 |
| L7_PROTO | 0.8253 |
| SRC_TO_DST_AVG_THROUGHPUT | 0.8013 |
| NUM_PKTS_512_TO_1024_BYTES | 0.7871 |
| CLIENT_TCP_FLAGS | 0.7436 |
| NUM_PKTS_256_TO_512_BYTES | 0.7372 |
| NUM_PKTS_128_TO_256_BYTES | 0.7305 |
| TCP_FLAGS | 0.7216 |
| DNS_QUERY_TYPE | 0.7147 |
| DNS_QUERY_ID | 0.7146 |
| PROTOCOL | 0.7127 |
| SERVER_TCP_FLAGS | 0.7120 |

Nghi lối tắt (AUC >= 0.9): không có

## Khoảng giá trị từng đặc trưng

| Đặc trưng | min | trung vị | p99 | max |
|---|---|---|---|---|
| L4_SRC_PORT | 0 | 54193 | 65309 | 65535 |
| L4_DST_PORT | 0 | 80 | 9200 | 65535 |
| PROTOCOL | 1 | 6 | 17 | 58 |
| L7_PROTO | 0 | 0 | 131 | 248 |
| IN_BYTES | 28 | 180 | 4181.01 | 1.65253e+08 |
| IN_PKTS | 1 | 4 | 35 | 292269 |
| OUT_BYTES | 0 | 328 | 33951.2 | 2.7564e+08 |
| OUT_PKTS | 0 | 3 | 49 | 184218 |
| TCP_FLAGS | 0 | 27 | 223 | 223 |
| CLIENT_TCP_FLAGS | 0 | 26 | 223 | 223 |
| SERVER_TCP_FLAGS | 0 | 22 | 94 | 223 |
| FLOW_DURATION_MILLISECONDS | 0 | 0 | 4.29496e+06 | 4.29497e+06 |
| DURATION_IN | 0 | 0 | 296 | 106714 |
| DURATION_OUT | 0 | 0 | 281 | 55477 |
| MIN_TTL | 0 | 100 | 128 | 255 |
| MAX_TTL | 0 | 101 | 128 | 255 |
| LONGEST_FLOW_PKT | 28 | 189 | 1500 | 65212 |
| SHORTEST_FLOW_PKT | 28 | 40 | 102 | 1500 |
| MIN_IP_PKT_LEN | 0 | 40 | 96 | 1260 |
| MAX_IP_PKT_LEN | 28 | 192 | 1500 | 65212 |
| SRC_TO_DST_SECOND_BYTES | 0 | 219 | 92431 | 6.58802e+304 |
| DST_TO_SRC_SECOND_BYTES | 0 | 338 | 489802 | 3.12209e+73 |
| RETRANSMITTED_IN_BYTES | 0 | 0 | 701 | 237032 |
| RETRANSMITTED_IN_PKTS | 0 | 0 | 11 | 1182 |
| RETRANSMITTED_OUT_BYTES | 0 | 0 | 1213 | 3.65681e+06 |
| RETRANSMITTED_OUT_PKTS | 0 | 0 | 2 | 2439 |
| SRC_TO_DST_AVG_THROUGHPUT | 0 | 1.008e+06 | 2.52e+07 | 3.84749e+09 |
| DST_TO_SRC_AVG_THROUGHPUT | 0 | 1.44e+06 | 1.48433e+08 | 4.29448e+09 |
| NUM_PKTS_UP_TO_128_BYTES | 0 | 6 | 48.01 | 292269 |
| NUM_PKTS_128_TO_256_BYTES | 0 | 1 | 7 | 8413 |
| NUM_PKTS_256_TO_512_BYTES | 0 | 0 | 6 | 4743 |
| NUM_PKTS_512_TO_1024_BYTES | 0 | 0 | 5 | 3988 |
| NUM_PKTS_1024_TO_1514_BYTES | 0 | 0 | 20 | 183631 |
| TCP_WIN_MAX_IN | 0 | 8192 | 65535 | 65535 |
| TCP_WIN_MAX_OUT | 0 | 8192 | 64800 | 65535 |
| ICMP_TYPE | 0 | 0 | 55554.6 | 65280 |
| ICMP_IPV4_TYPE | 0 | 0 | 216 | 255 |
| DNS_QUERY_ID | 0 | 0 | 63892 | 65535 |
| DNS_QUERY_TYPE | 0 | 0 | 28 | 4096 |
| DNS_TTL_ANSWER | 0 | 0 | 60 | 134473 |
| FTP_COMMAND_RET_CODE | 0 | 0 | 0 | 0 |

## Cặp đặc trưng tương quan cao (Spearman)

| Đặc trưng A | Đặc trưng B | rho |
|---|---|---|
| LONGEST_FLOW_PKT | MAX_IP_PKT_LEN | 1.0 |
| ICMP_TYPE | ICMP_IPV4_TYPE | 1.0 |
| RETRANSMITTED_OUT_BYTES | RETRANSMITTED_OUT_PKTS | 0.9997 |
| RETRANSMITTED_IN_BYTES | RETRANSMITTED_IN_PKTS | 0.9991 |
| RETRANSMITTED_IN_PKTS | ICMP_TYPE | 0.9943 |
| RETRANSMITTED_IN_PKTS | ICMP_IPV4_TYPE | 0.9943 |
| RETRANSMITTED_IN_BYTES | ICMP_TYPE | 0.9939 |
| RETRANSMITTED_IN_BYTES | ICMP_IPV4_TYPE | 0.9939 |
| OUT_BYTES | DST_TO_SRC_SECOND_BYTES | 0.9903 |
| TCP_FLAGS | CLIENT_TCP_FLAGS | 0.9832 |
| IN_BYTES | SRC_TO_DST_SECOND_BYTES | 0.9764 |
| FLOW_DURATION_MILLISECONDS | DURATION_IN | 0.9621 |
| PROTOCOL | DNS_QUERY_TYPE | 0.9614 |
| IN_PKTS | OUT_PKTS | 0.9531 |
| IN_PKTS | NUM_PKTS_UP_TO_128_BYTES | 0.9518 |
| OUT_BYTES | LONGEST_FLOW_PKT | 0.9514 |
| OUT_BYTES | MAX_IP_PKT_LEN | 0.9514 |
| PROTOCOL | DNS_QUERY_ID | 0.9442 |
| LONGEST_FLOW_PKT | DST_TO_SRC_SECOND_BYTES | 0.944 |
| MAX_IP_PKT_LEN | DST_TO_SRC_SECOND_BYTES | 0.944 |
| OUT_PKTS | NUM_PKTS_UP_TO_128_BYTES | 0.9433 |
| IN_BYTES | IN_PKTS | 0.9425 |
| PROTOCOL | DNS_TTL_ANSWER | 0.9405 |
| OUT_BYTES | OUT_PKTS | 0.9318 |
| DNS_QUERY_ID | DNS_QUERY_TYPE | 0.9314 |
| IN_PKTS | SRC_TO_DST_SECOND_BYTES | 0.9295 |
| OUT_PKTS | DST_TO_SRC_SECOND_BYTES | 0.9257 |
| PROTOCOL | SHORTEST_FLOW_PKT | 0.9251 |
| DNS_QUERY_TYPE | DNS_TTL_ANSWER | 0.9246 |
| FLOW_DURATION_MILLISECONDS | DURATION_OUT | 0.9228 |
| IN_BYTES | OUT_PKTS | 0.92 |
| SRC_TO_DST_SECOND_BYTES | NUM_PKTS_UP_TO_128_BYTES | 0.9149 |
| DURATION_IN | DURATION_OUT | 0.9123 |
| DNS_QUERY_ID | DNS_TTL_ANSWER | 0.9109 |
| OUT_PKTS | SRC_TO_DST_SECOND_BYTES | 0.9108 |
| TCP_FLAGS | SERVER_TCP_FLAGS | 0.9048 |
| PROTOCOL | TCP_WIN_MAX_IN | -0.9029 |
| SHORTEST_FLOW_PKT | MIN_IP_PKT_LEN | 0.9024 |
| IN_BYTES | NUM_PKTS_UP_TO_128_BYTES | 0.9007 |
