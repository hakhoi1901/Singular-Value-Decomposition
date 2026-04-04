"""
Định thức qua khử Gauss: det(A) = (-1)^s * prod U_ii.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_PART1 = Path(__file__).resolve().parent
for _dir in (_ROOT, _PART1):
    _s = str(_dir)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from gaussian import gaussian_eliminate


def determinant(A: list[list[float]]) -> float:
    """
    Tính det(A) sau khi đưa A về tam giác trên bằng khử Gauss (partial pivoting).
    Nếu ma trận suy biến, trả về 0.
    """
    if not A:
        return 1.0

    n = len(A)
    if any(len(row) != n for row in A):
        raise ValueError("A must be square")

    b = [0.0] * n
    try:
        U, _, swap_count = gaussian_eliminate(A, b)
    except ValueError:
        return 0.0

    det_a = 1.0 if (swap_count % 2 == 0) else -1.0
    for i in range(n):
        det_a *= U[i][i]
    return det_a
