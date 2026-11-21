"""Định dạng văn bản cho phần in kết quả: bảng căn cột, thanh tỷ lệ, tiêu đề.

Chỉ dùng thư viện chuẩn để mọi script đều gọi được. Ký tự khung dùng bộ Unicode phổ biến (─ │ ═ █ ░ ★),
cùng nhóm ký tự với "★ in-domain" mà các script cũ đã in ra được trên PowerShell.
"""
from __future__ import annotations

import math

WIDTH = 78


def num(v, nd: int = 3, sign: bool = False, dash: str = "-") -> str:
    """Số thực -> chuỗi cố định số chữ số thập phân; None/NaN -> dấu gạch."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return dash
    return f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"


def pct(v, nd: int = 0, dash: str = "-") -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return dash
    return f"{100 * v:.{nd}f}%"


def bar(v, vmax: float = 1.0, width: int = 12) -> str:
    """Thanh ngang █░ tỷ lệ với v/vmax (kẹp trong [0, 1])."""
    if v is None or (isinstance(v, float) and math.isnan(v)) or vmax <= 0:
        return "·" * width
    n = int(round(width * min(max(v / vmax, 0.0), 1.0)))
    return "█" * n + "░" * (width - n)


def banner(title: str, char: str = "═", width: int = WIDTH) -> str:
    return f"\n{char * width}\n {title}\n{char * width}"


def heading(title: str, width: int = WIDTH) -> str:
    return f"\n{title}\n{'─' * min(width, max(len(title), 20))}"


def table(headers: list[str], rows: list[list], aligns: str | None = None, indent: int = 2) -> str:
    """Bảng căn cột. aligns: chuỗi '<'/'>'/'^' mỗi ký tự cho một cột (mặc định: cột đầu trái, còn lại phải).

    Một hàng chỉ có một phần tử (list có 1 chuỗi) được in như dòng nhãn nhóm trải rộng cả bảng.
    """
    n = len(headers)
    aligns = (aligns or ("<" + ">" * (n - 1)))[:n].ljust(n, ">")
    cells = [[str(c) for c in r] if len(r) == n else [str(r[0])] for r in rows]
    widths = [len(str(h)) for h in headers]
    for r in cells:
        if len(r) == n:
            for i, c in enumerate(r):
                widths[i] = max(widths[i], len(c))

    def fmt(vals: list[str]) -> str:
        return " │ ".join(f"{v:{a}{w}}" for v, a, w in zip(vals, aligns, widths))

    pad = " " * indent
    total = sum(widths) + 3 * (n - 1)
    out = [pad + fmt([str(h) for h in headers]), pad + "─" * total]
    for r in cells:
        out.append(pad + (fmt(r) if len(r) == n else f"▸ {r[0]}"))
    return "\n".join(out)


def best_marks(values: list, higher_is_better: bool = True) -> list[bool]:
    """Đánh dấu các vị trí tốt nhất (cho phép nhiều vị trí bằng nhau, bỏ qua None/NaN)."""
    ok = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not ok:
        return [False] * len(values)
    target = max(ok) if higher_is_better else min(ok)
    return [v is not None and not (isinstance(v, float) and math.isnan(v)) and abs(v - target) < 1e-12 for v in values]
