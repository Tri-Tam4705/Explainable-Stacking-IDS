"""Tạo đặc trưng ứng viên từ các cột NetFlow đã làm sạch (hàm thuần, không học gì từ dữ liệu).

Mỗi đặc trưng thuộc một nhóm ngữ nghĩa cố định; nhóm là đơn vị giải thích ở các bước sau.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TCP_BITS = {"fin": 1, "syn": 2, "rst": 4, "psh": 8, "ack": 16, "urg": 32}
BIN_COLS = ["NUM_PKTS_UP_TO_128_BYTES", "NUM_PKTS_128_TO_256_BYTES", "NUM_PKTS_256_TO_512_BYTES",
            "NUM_PKTS_512_TO_1024_BYTES", "NUM_PKTS_1024_TO_1514_BYTES"]
BIN_NAMES = ["frac_pkts_le128", "frac_pkts_128_256", "frac_pkts_256_512", "frac_pkts_512_1024",
             "frac_pkts_1024_1514"]

GROUP_ORDER = ["volume", "timing_rate", "packet_size", "tcp_behavior", "retransmission", "protocol"]


def _div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros(len(a), dtype="float64")
    m = b > 0
    out[m] = a[m] / b[m]
    return out


def engineer(df: pd.DataFrame, extra: tuple[str, ...] = ()):
    """Trả về (X, groups): X là DataFrame float32, groups ánh xạ tên đặc trưng -> nhóm."""
    g = lambda c: df[c].to_numpy(dtype="float64")  # noqa: E731
    f: dict[str, np.ndarray] = {}
    groups: dict[str, str] = {}

    def put(name, values, group):
        f[name] = np.asarray(values, dtype="float64")
        groups[name] = group

    in_b, out_b, in_p, out_p = g("IN_BYTES"), g("OUT_BYTES"), g("IN_PKTS"), g("OUT_PKTS")
    put("in_bytes", in_b, "volume")
    put("out_bytes", out_b, "volume")
    put("in_pkts", in_p, "volume")
    put("out_pkts", out_p, "volume")
    put("in_byte_frac", _div(in_b, in_b + out_b), "volume")
    put("in_pkt_frac", _div(in_p, in_p + out_p), "volume")
    put("bytes_per_pkt_in", _div(in_b, in_p), "volume")
    put("bytes_per_pkt_out", _div(out_b, out_p), "volume")

    put("flow_duration_ms", g("FLOW_DURATION_MILLISECONDS"), "timing_rate")
    put("duration_in_ms", g("DURATION_IN"), "timing_rate")
    put("duration_out_ms", g("DURATION_OUT"), "timing_rate")
    put("src_to_dst_bytes_per_s", g("SRC_TO_DST_SECOND_BYTES"), "timing_rate")
    put("dst_to_src_bytes_per_s", g("DST_TO_SRC_SECOND_BYTES"), "timing_rate")
    put("src_to_dst_throughput", g("SRC_TO_DST_AVG_THROUGHPUT"), "timing_rate")
    put("dst_to_src_throughput", g("DST_TO_SRC_AVG_THROUGHPUT"), "timing_rate")

    put("longest_pkt", g("LONGEST_FLOW_PKT"), "packet_size")
    put("shortest_pkt", g("SHORTEST_FLOW_PKT"), "packet_size")
    put("max_ip_pkt_len", g("MAX_IP_PKT_LEN"), "packet_size")
    put("min_ip_pkt_len", g("MIN_IP_PKT_LEN"), "packet_size")
    bins = np.stack([g(c) for c in BIN_COLS], axis=1)
    tot = bins.sum(axis=1)
    for k, name in enumerate(BIN_NAMES):
        put(name, _div(bins[:, k], tot), "packet_size")

    for side, col in (("client", "CLIENT_TCP_FLAGS"), ("server", "SERVER_TCP_FLAGS")):
        v = g(col).astype("int64")
        for bit, mask in TCP_BITS.items():
            put(f"{side}_{bit}", ((v & mask) > 0).astype("float64"), "tcp_behavior")
    put("tcp_win_in", g("TCP_WIN_MAX_IN"), "tcp_behavior")
    put("tcp_win_out", g("TCP_WIN_MAX_OUT"), "tcp_behavior")

    put("retrans_in_pkts", g("RETRANSMITTED_IN_PKTS"), "retransmission")
    put("retrans_out_pkts", g("RETRANSMITTED_OUT_PKTS"), "retransmission")

    proto = g("PROTOCOL")
    put("is_tcp", proto == 6, "protocol")
    put("is_udp", proto == 17, "protocol")
    put("is_icmp", proto == 1, "protocol")
    put("is_other_proto", ~np.isin(proto, [1, 6, 17]), "protocol")

    if "L7_PROTO" in extra:  # chỉ dùng cho thí nghiệm cắt bỏ
        put("l7_proto", g("L7_PROTO"), "protocol")

    X = pd.DataFrame(f, index=df.index).astype("float32")
    return X, groups
