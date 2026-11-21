"""Lược đồ 43 đặc trưng NetFlow v2 (nProbe) và các nhóm ngữ nghĩa.

Nhóm ngữ nghĩa được cố định TRƯỚC khi nhìn dữ liệu. Chúng vừa dùng để lọc đặc
trưng (bước 2), vừa là đơn vị giải thích (cộng SHAP theo nhóm).
"""
from __future__ import annotations

# Tên cột chuẩn hóa nội bộ sau khi nạp dữ liệu
LABEL_COL = "label"      # 0 = bình thường, 1 = tấn công
ATTACK_COL = "attack"    # tên tấn công thô (Benign, DoS, ...)
FAMILY_COL = "family"    # họ đã gom (Benign, DoS, Other)
WEIGHT_COL = "n_orig"        # số dòng gốc có cùng vector đặc trưng (trọng số khôi phục tỷ lệ tự nhiên)
MINORITY_COL = "n_minority"  # số dòng trong nhóm đó có nhãn khác nhãn đại diện (nhiễu nhãn)

IDENTIFIER_COLS = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT", "L4_DST_PORT"]
TTL_COLS = ["MIN_TTL", "MAX_TTL"]

# Cột thời gian (nếu file có): chỉ giữ làm siêu dữ liệu, không đưa vào mô hình
TIME_META_COLS = ["FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS"]

FEATURE_GROUPS = {
    "identifier": IDENTIFIER_COLS,
    "ttl": TTL_COLS,
    "volume": ["IN_BYTES", "OUT_BYTES", "IN_PKTS", "OUT_PKTS"],
    "timing_rate": [
        "FLOW_DURATION_MILLISECONDS",
        "DURATION_IN",
        "DURATION_OUT",
        "SRC_TO_DST_SECOND_BYTES",
        "DST_TO_SRC_SECOND_BYTES",
        "SRC_TO_DST_AVG_THROUGHPUT",
        "DST_TO_SRC_AVG_THROUGHPUT",
    ],
    "packet_size": [
        "LONGEST_FLOW_PKT",
        "SHORTEST_FLOW_PKT",
        "MIN_IP_PKT_LEN",
        "MAX_IP_PKT_LEN",
        "NUM_PKTS_UP_TO_128_BYTES",
        "NUM_PKTS_128_TO_256_BYTES",
        "NUM_PKTS_256_TO_512_BYTES",
        "NUM_PKTS_512_TO_1024_BYTES",
        "NUM_PKTS_1024_TO_1514_BYTES",
    ],
    "tcp_behavior": [
        "TCP_FLAGS",
        "CLIENT_TCP_FLAGS",
        "SERVER_TCP_FLAGS",
        "TCP_WIN_MAX_IN",
        "TCP_WIN_MAX_OUT",
    ],
    "retransmission": [
        "RETRANSMITTED_IN_BYTES",
        "RETRANSMITTED_IN_PKTS",
        "RETRANSMITTED_OUT_BYTES",
        "RETRANSMITTED_OUT_PKTS",
    ],
    "protocol": ["PROTOCOL", "ICMP_TYPE", "ICMP_IPV4_TYPE"],
    "application": [
        "L7_PROTO",
        "DNS_QUERY_ID",
        "DNS_QUERY_TYPE",
        "DNS_TTL_ANSWER",
        "FTP_COMMAND_RET_CODE",
    ],
}

ALL_FEATURES = [c for cols in FEATURE_GROUPS.values() for c in cols]
IP_COLS = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR"]

assert len(ALL_FEATURES) == 43 and len(set(ALL_FEATURES)) == 43

# Cột bị loại ngay ở bước làm sạch (quy tắc cứng)
HARD_DROP_COLS = IDENTIFIER_COLS + TTL_COLS

