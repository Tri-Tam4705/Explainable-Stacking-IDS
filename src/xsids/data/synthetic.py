"""Sinh dữ liệu giả lập có lược đồ NetFlow v2 để kiểm thử pipeline khi chưa có dữ liệu thật.

KHÔNG dùng để rút ra kết luận khoa học. Lối tắt TTL được cài chủ ý rất mạnh (máy tấn công
có TTL ~254 gần như cố định) để kiểm tra khâu kiểm toán. Dữ liệu cố ý chứa các lỗi giống thực tế
(dòng trùng, inf/NaN, xung đột nhãn) và các đặc trưng "lối tắt" (TTL, cổng) để
kiểm tra khâu kiểm toán và làm sạch.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from xsids.schema import ALL_FEATURES

# (tên lớp, xác suất) cho từng kiểu bộ dữ liệu
_CLASS_MIX = {
    "unsw": [("Benign", 0.9602), ("Exploits", 0.0120), ("Generic", 0.0080), ("Fuzzers", 0.0060),
             ("Reconnaissance", 0.0050), ("DoS", 0.0035), ("Backdoor", 0.0020),
             ("Analysis", 0.0018), ("Shellcode", 0.0010), ("Worms", 0.0005)],
    "cse": [("Benign", 0.8805), ("DDoS", 0.0735), ("DoS", 0.0256), ("Bot", 0.0063),
            ("Brute Force -Web", 0.0050), ("SSH-Bruteforce", 0.0040), ("Infilteration", 0.0030),
            ("SQL Injection", 0.0015), ("Brute Force -XSS", 0.0006)],
}

_BASE = dict(
    proto_p=(0.80, 0.18, 0.02), in_pkts=(1.6, 1.0), out_pkts=(1.5, 1.0), size=(5.4, 0.9),
    dur=(7.0, 2.2), cflags=[27, 19, 24, 2], sflags=[27, 18, 22, 20],
    bin_p=(0.55, 0.12, 0.08, 0.10, 0.15), ttl=(64, 128), ttl_p=(0.6, 0.4),
    dport=[80, 443, 53, 22, 8080, 25],
)
_DOS = dict(_BASE, proto_p=(0.90, 0.10, 0.0), in_pkts=(2.2, 0.6), out_pkts=(0.2, 0.5), size=(4.3, 0.3),
            dur=(4.5, 1.5), cflags=[2, 2, 2, 27], sflags=[0, 0, 20, 18],
            bin_p=(0.95, 0.02, 0.01, 0.01, 0.01), ttl=(254, 62), ttl_p=(0.96, 0.04), dport=[80, 80, 443])
_RECON = dict(_BASE, in_pkts=(0.2, 0.4), out_pkts=(0.0, 0.3), size=(4.0, 0.2), dur=(2.0, 1.2),
              cflags=[2], sflags=[20, 0, 18], ttl=(254, 62), ttl_p=(0.96, 0.04),
              bin_p=(0.97, 0.01, 0.01, 0.005, 0.005), dport=[21, 22, 23, 80, 445, 3389])
# các tấn công còn lại: gần với lưu lượng bình thường (lớp "khó")
_HARD = dict(_BASE, size=(5.1, 0.9), dur=(6.3, 2.2), ttl=(254, 64), ttl_p=(0.96, 0.04),
             dport=[21, 22, 80, 445, 1433, 3306])


def _spec(name: str) -> dict:
    n = name.lower()
    if n == "benign":
        return _BASE
    if n in ("dos", "ddos"):
        return _DOS
    if n.startswith("recon"):
        return _RECON
    return _HARD


def _block(rng: np.random.Generator, m: int, sp: dict) -> dict:
    proto = rng.choice([6, 17, 1], size=m, p=sp["proto_p"])
    in_pkts = np.maximum(1, rng.lognormal(*sp["in_pkts"], m)).astype(np.int64)
    out_pkts = np.maximum(0, rng.lognormal(*sp["out_pkts"], m) - 0.5).astype(np.int64)
    size_in = np.clip(rng.lognormal(*sp["size"], m), 40, 1500)
    size_out = np.clip(rng.lognormal(sp["size"][0] + 0.3, sp["size"][1], m), 40, 1500)
    in_bytes = (in_pkts * size_in).astype(np.int64)
    out_bytes = (out_pkts * size_out).astype(np.int64)
    dur_ms = np.clip(rng.lognormal(*sp["dur"], m), 0, 3.6e6).astype(np.int64)
    dur_in = (dur_ms * rng.uniform(0.3, 1.0, m)).astype(np.int64)
    dur_out = (dur_ms * rng.uniform(0.0, 1.0, m)).astype(np.int64)
    sec = np.maximum(dur_ms / 1000.0, 0.001)
    s2d = in_bytes / sec
    d2s = out_bytes / sec

    longest = np.clip(size_in * 1.4, 40, 1514).astype(np.int64)
    shortest = np.clip(size_in * 0.4, 40, 1514).astype(np.int64)
    shortest = np.minimum(shortest, longest)
    total_pkts = in_pkts + out_pkts
    bins = rng.multinomial(total_pkts, np.array(sp["bin_p"]) / np.sum(sp["bin_p"]))

    is_tcp = proto == 6
    cflags = np.where(is_tcp, rng.choice(sp["cflags"], m), 0)
    sflags = np.where(is_tcp, rng.choice(sp["sflags"], m), 0)
    win_in = np.where(is_tcp, rng.choice([0, 8192, 29200, 65535], m), 0)
    win_out = np.where(is_tcp, rng.choice([0, 8192, 29200, 65535], m), 0)
    retr_in = np.where(is_tcp, rng.poisson(0.3, m), 0)
    retr_out = np.where(is_tcp, rng.poisson(0.2, m), 0)

    is_icmp = proto == 1
    # mô phỏng lỗi thật của NF-v2: luồng không phải ICMP mang giá trị của bộ đếm truyền lại
    icmp_ipv4 = np.where(is_icmp, rng.choice([8, 0, 3], m), (retr_in * 60) % 256)
    icmp_type = icmp_ipv4 * 256
    l7 = rng.choice([0, 7, 5, 91, 92], m, p=[0.3, 0.2, 0.15, 0.3, 0.05])
    is_dns = l7 == 5

    ttl_choice = rng.choice(sp["ttl"], m, p=sp["ttl_p"])
    return {
        "L4_SRC_PORT": rng.integers(1024, 65535, m),
        "L4_DST_PORT": rng.choice(sp["dport"], m),
        "PROTOCOL": proto, "L7_PROTO": l7,
        "IN_BYTES": in_bytes, "OUT_BYTES": out_bytes, "IN_PKTS": in_pkts, "OUT_PKTS": out_pkts,
        "FLOW_DURATION_MILLISECONDS": dur_ms,
        "TCP_FLAGS": cflags | sflags, "CLIENT_TCP_FLAGS": cflags, "SERVER_TCP_FLAGS": sflags,
        "DURATION_IN": dur_in, "DURATION_OUT": dur_out,
        "MIN_TTL": ttl_choice, "MAX_TTL": ttl_choice + rng.integers(0, 3, m),
        "LONGEST_FLOW_PKT": longest, "SHORTEST_FLOW_PKT": shortest,
        "MIN_IP_PKT_LEN": shortest, "MAX_IP_PKT_LEN": longest,
        "SRC_TO_DST_SECOND_BYTES": s2d, "DST_TO_SRC_SECOND_BYTES": d2s,
        "RETRANSMITTED_IN_BYTES": retr_in * 60, "RETRANSMITTED_IN_PKTS": retr_in,
        "RETRANSMITTED_OUT_BYTES": retr_out * 60, "RETRANSMITTED_OUT_PKTS": retr_out,
        "SRC_TO_DST_AVG_THROUGHPUT": np.round(s2d * 8), "DST_TO_SRC_AVG_THROUGHPUT": np.round(d2s * 8),
        "NUM_PKTS_UP_TO_128_BYTES": bins[:, 0], "NUM_PKTS_128_TO_256_BYTES": bins[:, 1],
        "NUM_PKTS_256_TO_512_BYTES": bins[:, 2], "NUM_PKTS_512_TO_1024_BYTES": bins[:, 3],
        "NUM_PKTS_1024_TO_1514_BYTES": bins[:, 4],
        "TCP_WIN_MAX_IN": win_in, "TCP_WIN_MAX_OUT": win_out,
        "ICMP_TYPE": icmp_type, "ICMP_IPV4_TYPE": icmp_ipv4,
        "DNS_QUERY_ID": np.where(is_dns, rng.integers(1, 65535, m), 0),
        "DNS_QUERY_TYPE": np.where(is_dns, 1, 0),
        "DNS_TTL_ANSWER": np.where(is_dns, rng.integers(30, 3600, m), 0),
        "FTP_COMMAND_RET_CODE": np.where(rng.random(m) < 0.01, 220, 0),
    }


def make_synthetic_netflow(n: int = 50_000, style: str = "unsw", seed: int = 0, defects: bool = True,
                           include_ip: bool = True, include_time: bool = True) -> pd.DataFrame:
    """Trả về DataFrame có 43 đặc trưng + Label + Attack (+ IP và thời gian nếu bật)."""
    rng = np.random.default_rng(seed)
    names = [c for c, _ in _CLASS_MIX[style]]
    p = np.array([q for _, q in _CLASS_MIX[style]], dtype=float)
    cls = rng.choice(len(names), size=n, p=p / p.sum())

    parts = []
    for k, name in enumerate(names):
        idx = np.where(cls == k)[0]
        if len(idx) == 0:
            continue
        blk = pd.DataFrame(_block(rng, len(idx), _spec(name)), index=idx)
        blk["Attack"] = name
        blk["Label"] = 0 if name == "Benign" else 1
        parts.append(blk)
    df = pd.concat(parts).sort_index()

    if include_ip:
        df["IPV4_SRC_ADDR"] = [f"10.0.{a}.{b}" for a, b in zip(rng.integers(0, 255, n), rng.integers(1, 255, n))]
        df["IPV4_DST_ADDR"] = [f"192.168.{a}.{b}" for a, b in zip(rng.integers(0, 255, n), rng.integers(1, 255, n))]
    if include_time:
        start = np.sort(rng.integers(1_500_000_000_000, 1_500_000_000_000 + 2_000_000_000, n))
        df["FLOW_START_MILLISECONDS"] = start
        df["FLOW_END_MILLISECONDS"] = start + df["FLOW_DURATION_MILLISECONDS"].to_numpy()

    if defects:
        df = _inject_defects(df, rng)

    cols = [c for c in ["IPV4_SRC_ADDR", "L4_SRC_PORT", "IPV4_DST_ADDR", "L4_DST_PORT"] if c in df.columns]
    cols += [c for c in ALL_FEATURES if c in df.columns and c not in cols]
    cols += [c for c in ["FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS"] if c in df.columns]
    cols += ["Label", "Attack"]
    return df[cols].reset_index(drop=True)


def _inject_defects(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    n = len(df)
    df = df.copy()
    # 1) giá trị không hữu hạn
    i_inf = rng.choice(n, size=max(1, int(0.0005 * n)), replace=False)
    i_nan = rng.choice(n, size=max(1, int(0.0003 * n)), replace=False)
    df["SRC_TO_DST_SECOND_BYTES"] = df["SRC_TO_DST_SECOND_BYTES"].astype("float64")
    df["DST_TO_SRC_SECOND_BYTES"] = df["DST_TO_SRC_SECOND_BYTES"].astype("float64")
    df.iloc[i_inf, df.columns.get_loc("SRC_TO_DST_SECOND_BYTES")] = np.inf
    df.iloc[i_nan, df.columns.get_loc("DST_TO_SRC_SECOND_BYTES")] = np.nan
    # 1b) giá trị đệm kiểu DBL_MAX ở một cột số thực (khoảng 0,05%)
    i_big = rng.choice(n, size=max(2, int(0.0005 * n)), replace=False)
    df["DST_TO_SRC_AVG_THROUGHPUT"] = df["DST_TO_SRC_AVG_THROUGHPUT"].astype("float64")
    df.iloc[i_big, df.columns.get_loc("DST_TO_SRC_AVG_THROUGHPUT")] = 1.7976931348623157e308
    # 2) dòng trùng lặp hoàn toàn (khoảng 3%)
    dup = df.iloc[rng.choice(n, size=int(0.03 * n), replace=False)]
    # 3) xung đột nhãn: bản sao có nhãn bị đảo (khoảng 0,2%)
    conf = df.iloc[rng.choice(n, size=max(2, int(0.002 * n)), replace=False)].copy()
    atk = [a for a in df["Attack"].unique() if a != "Benign"][0]
    flip_to_attack = conf["Label"].to_numpy() == 0
    conf["Attack"] = np.where(flip_to_attack, atk, "Benign")
    conf["Label"] = np.where(flip_to_attack, 1, 0)
    return pd.concat([df, dup, conf], ignore_index=True)
