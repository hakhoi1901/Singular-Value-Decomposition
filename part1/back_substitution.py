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


def back_substitution(U: list[list[float]], c: list[float]) -> list[float]:
    """
    Giải hệ phương trình tam giác trên Ux = c từ dưới lên.

    Tham số:
    - U: Ma trận tam giác trên vuông kích thước n x n
    - c: Vector vế phải kích thước n
    """
    if not U:
        return []

    n = len(U)
    if any(len(row) != n for row in U):
        raise ValueError("Matrix U must be square (n x n)")
    if len(c) != n:
        raise ValueError("Vector c must have the same length as U")

    x = [0.0] * n

    for i in range(n - 1, -1, -1):
        if is_zero(U[i][i]):
            raise ValueError("Matrix is singular")

        s = c[i]
        for j in range(i + 1, n):
            s -= U[i][j] * x[j]
        x[i] = s / U[i][i]

    return x
