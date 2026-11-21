# Báo cáo kiểm toán dữ liệu: NF-UNSW-NB15-v2

- Số dòng: **2,390,275**, số cột: 44, bộ nhớ: 854.0 MB
- Bình thường: 2,295,222 | Tấn công: 95,053 | Tỷ lệ tấn công: **3.98%**

## Phân bố lớp tấn công

| Lớp | Số luồng |
|---|---|
| Benign | 2,295,222 |
| Exploits | 31,551 |
| Fuzzers | 22,310 |
| Generic | 16,560 |
| Reconnaissance | 12,779 |
| DoS | 5,794 |
| Analysis | 2,299 |
| Backdoor | 2,169 |
| Shellcode | 1,427 |
| Worms | 164 |

- Tên tấn công không khớp mẫu họ nào (vào *Other*): ['Analysis', 'Backdoor', 'Exploits', 'Fuzzers', 'Generic', 'Reconnaissance', 'Shellcode', 'Worms']
- Dòng có nhãn nhị phân mâu thuẫn với họ Benign: 0

## Chất lượng dữ liệu

- Dòng có giá trị không hữu hạn (inf/NaN): 0 {}
- Cột có giá trị âm: không có
- Cột hằng (phương sai 0): không có
- Cột thưa (tỷ lệ giá trị 0 >= ngưỡng): không có
- Dòng trùng lặp (thô): 403,530
- Dòng trùng lặp sau khi bỏ định danh và TTL: 2,149,110
- Xung đột nhãn (cùng đặc trưng, khác nhãn): 42 dòng, 17 nhóm
- Vector đặc trưng duy nhất sau khi bỏ định danh và TTL: 241,165 (10.1% số dòng)
- Giá trị |x| vượt ngưỡng vật lý (nghi giá trị đệm): {'SRC_TO_DST_SECOND_BYTES': 1}
- Trường ICMP trên luồng không phải ICMP: 1,691,887/2,388,719 dòng có ICMP_IPV4_TYPE khác 0, trong đó 202,290 dòng bằng RETRANSMITTED_IN_BYTES

## Đặc trưng có nhiều dòng sát giá trị lớn nhất (bão hòa hoặc tràn số)

| Đặc trưng | max | Tỷ lệ dòng | Tỷ lệ tấn công trong nhóm | Tỷ lệ tấn công chung |
|---|---|---|---|---|
| L4_SRC_PORT | 65535 | 0.12% | 17.9% | 4.0% |
| FLOW_DURATION_MILLISECONDS | 4.29496e+06 | 6.03% | 6.9% | 4.0% |
| MAX_TTL | 255 | 1.88% | 80.1% | 4.0% |

## Đặc trưng nghi là lối tắt (AUC đơn đặc trưng)

| Đặc trưng | AUC |
|---|---|
| MIN_TTL | 0.9946 |
| MAX_TTL | 0.9941 |
| MIN_IP_PKT_LEN | 0.9891 |
| SERVER_TCP_FLAGS | 0.7991 |
| DST_TO_SRC_AVG_THROUGHPUT | 0.7935 |
| OUT_BYTES | 0.7823 |
| DST_TO_SRC_SECOND_BYTES | 0.7818 |
| OUT_PKTS | 0.7776 |
| NUM_PKTS_UP_TO_128_BYTES | 0.7633 |
| RETRANSMITTED_OUT_PKTS | 0.7486 |
| RETRANSMITTED_OUT_BYTES | 0.7447 |
| TCP_FLAGS | 0.7256 |
| CLIENT_TCP_FLAGS | 0.7134 |
| IN_PKTS | 0.7065 |
| SRC_TO_DST_AVG_THROUGHPUT | 0.6891 |

Nghi lối tắt (AUC >= 0.9): ['MIN_TTL', 'MAX_TTL', 'MIN_IP_PKT_LEN']

## Khoảng giá trị từng đặc trưng

| Đặc trưng | min | trung vị | p99 | max |
|---|---|---|---|---|
| L4_SRC_PORT | 0 | 32716 | 64917 | 65535 |
| L4_DST_PORT | 0 | 80 | 63626 | 65535 |
| PROTOCOL | 0 | 6 | 17 | 255 |
| L7_PROTO | 0 | 0 | 92 | 248 |
| IN_BYTES | 1 | 1684 | 37466 | 3.02414e+07 |
| IN_PKTS | 1 | 18 | 358 | 22894 |
| OUT_BYTES | 0 | 2456 | 1.08789e+06 | 1.46585e+07 |
| OUT_PKTS | 0 | 18 | 746 | 11078 |
| TCP_FLAGS | 0 | 27 | 27 | 31 |
| CLIENT_TCP_FLAGS | 0 | 27 | 27 | 31 |
| SERVER_TCP_FLAGS | 0 | 27 | 27 | 31 |
| FLOW_DURATION_MILLISECONDS | 0 | 0 | 4.29495e+06 | 4.29496e+06 |
| DURATION_IN | 0 | 0 | 16 | 1047 |
| DURATION_OUT | 0 | 0 | 16 | 875 |
| MIN_TTL | 0 | 31 | 254 | 255 |
| MAX_TTL | 0 | 32 | 255 | 255 |
| LONGEST_FLOW_PKT | 28 | 485 | 1500 | 1508 |
| SHORTEST_FLOW_PKT | 28 | 52 | 100 | 1508 |
| MIN_IP_PKT_LEN | 0 | 52 | 76 | 680 |
| MAX_IP_PKT_LEN | 28 | 487 | 1500 | 1508 |
| SRC_TO_DST_SECOND_BYTES | 52 | 1698 | 37500 | 1.57282e+15 |
| DST_TO_SRC_SECOND_BYTES | 0 | 2456 | 1.08789e+06 | 2.61706e+14 |
| RETRANSMITTED_IN_BYTES | 0 | 506 | 18259 | 1.51205e+07 |
| RETRANSMITTED_IN_PKTS | 0 | 3 | 34 | 11442 |
| RETRANSMITTED_OUT_BYTES | 0 | 871 | 543733 | 7.32914e+06 |
| RETRANSMITTED_OUT_PKTS | 0 | 4 | 369 | 5536 |
| SRC_TO_DST_AVG_THROUGHPUT | 0 | 1.134e+07 | 2.99424e+08 | 4.27741e+09 |
| DST_TO_SRC_AVG_THROUGHPUT | 0 | 1.3824e+07 | 7.18848e+08 | 4.2939e+09 |
| NUM_PKTS_UP_TO_128_BYTES | 0 | 32 | 370 | 4824 |
| NUM_PKTS_128_TO_256_BYTES | 0 | 0 | 18 | 298 |
| NUM_PKTS_256_TO_512_BYTES | 0 | 0 | 18 | 3928 |
| NUM_PKTS_512_TO_1024_BYTES | 0 | 0 | 32 | 800 |
| NUM_PKTS_1024_TO_1514_BYTES | 0 | 0 | 704 | 19676 |
| TCP_WIN_MAX_IN | 0 | 14480 | 65160 | 65533 |
| TCP_WIN_MAX_OUT | 0 | 13032 | 55024 | 65531 |
| ICMP_TYPE | 0 | 11008 | 64256 | 65317 |
| ICMP_IPV4_TYPE | 0 | 43 | 251 | 255 |
| DNS_QUERY_ID | 0 | 0 | 60866 | 65535 |
| DNS_QUERY_TYPE | 0 | 0 | 1 | 55937 |
| DNS_TTL_ANSWER | 0 | 0 | 60 | 4.28336e+09 |
| FTP_COMMAND_RET_CODE | 0 | 0 | 331 | 501 |

## Cặp đặc trưng tương quan cao (Spearman)

| Đặc trưng A | Đặc trưng B | rho |
|---|---|---|
| LONGEST_FLOW_PKT | MAX_IP_PKT_LEN | 1.0 |
| ICMP_TYPE | ICMP_IPV4_TYPE | 1.0 |
| OUT_BYTES | DST_TO_SRC_SECOND_BYTES | 0.9971 |
| DNS_QUERY_ID | DNS_QUERY_TYPE | 0.9939 |
| FLOW_DURATION_MILLISECONDS | DURATION_IN | 0.9891 |
| IN_BYTES | SRC_TO_DST_SECOND_BYTES | 0.986 |
| OUT_BYTES | RETRANSMITTED_OUT_BYTES | 0.9845 |
| DST_TO_SRC_SECOND_BYTES | RETRANSMITTED_OUT_BYTES | 0.9831 |
| OUT_PKTS | RETRANSMITTED_OUT_PKTS | 0.9809 |
| IN_PKTS | OUT_PKTS | 0.9804 |
| FLOW_DURATION_MILLISECONDS | DURATION_OUT | 0.9802 |
| DNS_QUERY_ID | DNS_TTL_ANSWER | 0.9772 |
| DNS_QUERY_TYPE | DNS_TTL_ANSWER | 0.975 |
| DURATION_IN | DURATION_OUT | 0.9729 |
| OUT_PKTS | NUM_PKTS_UP_TO_128_BYTES | 0.964 |
| RETRANSMITTED_OUT_PKTS | TCP_WIN_MAX_IN | 0.9598 |
| TCP_FLAGS | SERVER_TCP_FLAGS | 0.9575 |
| IN_PKTS | RETRANSMITTED_OUT_PKTS | 0.956 |
| OUT_PKTS | TCP_WIN_MAX_IN | 0.9529 |
| OUT_BYTES | DST_TO_SRC_AVG_THROUGHPUT | 0.951 |
| DST_TO_SRC_SECOND_BYTES | DST_TO_SRC_AVG_THROUGHPUT | 0.9486 |
| IN_PKTS | NUM_PKTS_UP_TO_128_BYTES | 0.9472 |
| IN_PKTS | TCP_WIN_MAX_IN | 0.9458 |
| RETRANSMITTED_OUT_PKTS | NUM_PKTS_UP_TO_128_BYTES | 0.9455 |
| RETRANSMITTED_IN_PKTS | TCP_WIN_MAX_OUT | 0.9429 |
| RETRANSMITTED_OUT_BYTES | DST_TO_SRC_AVG_THROUGHPUT | 0.9425 |
| TCP_FLAGS | CLIENT_TCP_FLAGS | 0.9409 |
| NUM_PKTS_UP_TO_128_BYTES | TCP_WIN_MAX_IN | 0.9367 |
| DST_TO_SRC_SECOND_BYTES | RETRANSMITTED_OUT_PKTS | 0.9312 |
| RETRANSMITTED_OUT_BYTES | RETRANSMITTED_OUT_PKTS | 0.931 |
| OUT_BYTES | RETRANSMITTED_OUT_PKTS | 0.9285 |
| OUT_PKTS | DST_TO_SRC_SECOND_BYTES | 0.9279 |
| OUT_BYTES | OUT_PKTS | 0.9277 |
| IN_PKTS | SRC_TO_DST_SECOND_BYTES | 0.9193 |
| IN_BYTES | IN_PKTS | 0.9159 |
| IN_PKTS | RETRANSMITTED_IN_PKTS | 0.9104 |
| OUT_PKTS | RETRANSMITTED_OUT_BYTES | 0.9027 |
