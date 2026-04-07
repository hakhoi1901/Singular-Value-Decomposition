"""
Thế ngược cho hệ tam giác trên: ma trận tăng cường [U | c].
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import is_zero


def back_substitution(U: list[list[float]]) -> list[float]:
    """
    Giải Ux = c từ dưới lên.

    Tham số U là ma trận tăng cường: mỗi dòng có dạng
    (u_i0, u_i1, …, u_{i,n-1}, c_i), tức n cột hệ số tam giác trên + 1 cột vế phải.
    """
    if not U:
        return []

    n = len(U)
    width = len(U[0])
    if width != n + 1:
        raise ValueError("Each row of U must have length n+1 (augmented [coefficients | RHS])")
    if any(len(row) != width for row in U):
        raise ValueError("All rows of U must have the same length")

    x = [0.0] * n

    for i in range(n - 1, -1, -1):
        if is_zero(U[i][i]):
            raise ValueError("Matrix is singular")

        s = U[i][n]
        for j in range(i + 1, n):
            s -= U[i][j] * x[j]
        x[i] = s / U[i][i]

    return x
