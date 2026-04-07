"""
Thế ngược: giải Ux = c với U vuông tam giác trên (nghiệm duy nhất khi U khả nghịch).

Nếu gặp phần tử đường chéo ~0 (không thể thế ngược một cách duy nhất):
in "Hệ không có nghiệm duy nhất" và trả về [].
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
    - U: Ma trận tam giác trên vuông kích thước n × n
    - c: Vector vế phải kích thước n

    Khi U[i][i] ≈ 0: in cảnh báo và trả về [] (không dùng raise).
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
            print("Hệ không có nghiệm duy nhất")
            return []

        s = c[i]
        for j in range(i + 1, n):
            s -= U[i][j] * x[j]
        x[i] = s / U[i][i]

    return x


def test_back_substitution():
    import os
    import sys

    _parent = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if _parent not in sys.path:
        sys.path.insert(_parent)

    from verify_solution import verify_solution

    test_cases = [
        {
            "name": "2x2 nghiem nguyen",
            "U": [[1.0, 2.0], [0.0, 3.0]],
            "c": [7.0, 9.0],
            "expect_x": [1.0, 3.0],
        },
        {
            "name": "3x3 tam giac tren day du",
            "U": [[2.0, -1.0, 3.0], [0.0, 4.0, 1.0], [0.0, 0.0, 5.0]],
            "c": [9.0, 11.0, 15.0],
            "expect_x": [1.0, 2.0, 3.0],
        },
        {
            "name": "Don vi 4x4",
            "U": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "c": [1.0, -2.0, 3.0, -4.0],
            "expect_x": [1.0, -2.0, 3.0, -4.0],
        },
        {
            "name": "1x1",
            "U": [[8.0]],
            "c": [24.0],
            "expect_x": [3.0],
        },
        {
            "name": "Cheo lon (scale)",
            "U": [[1.0e6, 2.0], [0.0, 1.0e6]],
            "c": [-5.0e6 + 6.0, 3.0e6],
            "expect_x": [-5.0, 3.0],
        },
        {
            "name": "Duong cheo suy bien — tra ve rong",
            "U": [[1.0, 0.0], [0.0, 0.0]],
            "c": [1.0, 0.0],
            "expect_x": [],
        },
        {
            "name": "c sai do dai",
            "U": [[1.0, 0.0], [0.0, 1.0]],
            "c": [1.0],
            "should_raise": ValueError,
        },
        {
            "name": "U khong vuong (hang lech)",
            "U": [[1.0, 2.0, 3.0]],
            "c": [1.0],
            "should_raise": ValueError,
        },
        {
            "name": "Ma tran rong",
            "U": [],
            "c": [],
            "expect_x": [],
        },
    ]

    for case in test_cases:
        print(f"[back_substitution] {case['name']}")
        try:
            x = back_substitution(case["U"], case["c"])
            if case.get("should_raise"):
                assert False, "expected ValueError"
            if "expect_x" in case:
                assert len(x) == len(case["expect_x"])
                for i, v in enumerate(case["expect_x"]):
                    assert abs(x[i] - v) < 1e-6, f"x[{i}]={x[i]}"
            n = len(case["U"])
            if n > 0 and len(case["c"]) == n and x:
                e = verify_solution(case["U"], x, case["c"])
                assert e < 1e-8, f"||c-Ux||={e}"
            print("  PASSED")
        except ValueError as err:
            if case.get("should_raise") == ValueError:
                print(f"  PASSED ({err})")
            else:
                raise


if __name__ == "__main__":
    test_back_substitution()
