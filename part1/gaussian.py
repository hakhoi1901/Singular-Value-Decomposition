"""
Khử Gauss với partial pivoting cho hệ Ax = b (A vuông).
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import is_zero, zero_rectify


def gaussian_eliminate(A: list[list[float]], b: list[float]):
    """
    Đưa [A | b] về dạng tam giác trên bằng khử Gauss có chọn phần tử chốt cột.

    Returns:
        U: ma trận tam giác trên (n x n)
        c: vector vế phải sau biến đổi
        swap_count: số lần hoán đổi dòng
    """
    if not A:
        raise ValueError("Matrix is singular")

    n = len(A)
    if len(b) != n:
        raise ValueError("Vector b must have same length as number of rows in A")

    m = len(A[0])
    if any(len(row) != m for row in A):
        raise ValueError("All rows of A must have the same length")
    if m != n:
        raise ValueError("A must be square for this implementation")

    M = [list(row) + [float(b[i])] for i, row in enumerate(A)]
    swap_count = 0

    for k in range(n):
        p = k
        best = abs(M[k][k])
        for i in range(k + 1, n):
            v = abs(M[i][k])
            if v > best:
                best = v
                p = i

        pivot = M[p][k]

        if is_zero(pivot):
            raise ValueError("Matrix is singular")

        # Pivot còn nhỏ so với thang đo thông thường → có thể kém ổn định (bổ sung cho điều |M_pk| < ε trong đề)
        if abs(pivot) < 1e-8:
            warnings.warn(
                "Pivot is very small; the system may be ill-conditioned.",
                UserWarning,
                stacklevel=2,
            )

        if p != k:
            M[k], M[p] = M[p], M[k]
            swap_count += 1

        pivot = M[k][k]
        for i in range(k + 1, n):
            if is_zero(M[i][k]):
                continue
            lik = M[i][k] / pivot
            for j in range(k, n + 1):
                M[i][j] -= lik * M[k][j]
                M[i][j] = zero_rectify(M[i][j])

    U = [row[:n] for row in M]
    c = [row[n] for row in M]
    return U, c, swap_count
