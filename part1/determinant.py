"""
Định thức: chỉ xét A vuông n × n (đề bài).

det(A) = (-1)^s * ∏_i u_ii sau khi khử Gauss (partial pivot) đưa A về tam giác trên,
không bỏ qua cột — nếu không có pivot tại cột k thì det = 0.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import is_zero, zero_rectify
# IMPORT HÀM GAUSS ĐÃ VIẾT (Tận dụng code cũ)
from gaussian import gaussian_eliminate


def determinant(A: list[list[float]]) -> float:
    """
    Tính det(A) với A ∈ R^(n×n) bằng khử Gauss có chọn phần tử chốt.
    Sử dụng lại hàm gaussian_eliminate đã được test kỹ lưỡng.
    """
    if not A:
        return 1.0

    n = len(A)
    if any(len(row) != n for row in A):
        raise ValueError("A phải vuông (n × n) để tính định thức")

    # Tạo vector b giả (toàn số 0) để đưa vào hàm gaussian_eliminate
    b_dummy = [0.0] * n
    
    # Dùng hàm khử Gauss chuẩn của nhóm
    U, _, swap_count = gaussian_eliminate(A, b_dummy)

    # Tính tích đường chéo
    det_a = 1.0
    for i in range(n):
        det_a *= U[i][i]
        
    # Áp dụng dấu hoán vị
    if swap_count % 2 != 0:
        det_a = -det_a
        
    return zero_rectify(det_a)


def test_determinant():
    import warnings
    warnings.simplefilter("ignore", UserWarning) # Bỏ qua warning pivot nhỏ
    
    test_cases = [
        {
            "name": "Don vi 2x2",
            "A": [[1.0, 0.0], [0.0, 1.0]],
            "expected": 1.0,
        },
        {
            "name": "2x2 [[1,2],[3,4]]",
            "A": [[1.0, 2.0], [3.0, 4.0]],
            "expected": -2.0,
        },
        {
            "name": "2x2 [[2,1],[1,3]] (det=5)",
            "A": [[2.0, 1.0], [1.0, 3.0]],
            "expected": 5.0,
        },
        {
            "name": "Hoan vi [[0,1],[1,0]]",
            "A": [[0.0, 1.0], [1.0, 0.0]],
            "expected": -1.0,
        },
        {
            "name": "Suy bien [[1,2],[2,4]]",
            "A": [[1.0, 2.0], [2.0, 4.0]],
            "expected": 0.0,
        },
        {
            "name": "Tam giac tren — tich duong cheo",
            "A": [[2.0, 5.0, 1.0], [0.0, 3.0, 4.0], [0.0, 0.0, 6.0]],
            "expected": 36.0,
        },
        {
            "name": "3x3 inverse test",
            "A": [[1.0, 0.0, 5.0], [2.0, 1.0, 6.0], [3.0, 4.0, 0.0]],
            "expected": 1.0,
        },
        {
            "name": "Cheo diag(1,2,3,4)",
            "A": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0, 0.0],
                [0.0, 0.0, 3.0, 0.0],
                [0.0, 0.0, 0.0, 4.0],
            ],
            "expected": 24.0,
        },
        {
            "name": "1x1",
            "A": [[-3.5]],
            "expected": -3.5,
        },
        {
            "name": "Toan khong 3x3",
            "A": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            "expected": 0.0,
        },
        {
            "name": "Hang lap — rank 1",
            "A": [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
            "expected": 0.0,
        },
        {
            "name": "Ma trận 3x3 (Đã sửa lại Toán: = 2.0)",
            "A": [[1, 2, 3], [0, 4, 5], [1, 0, 1]],
            "expected": 2.0,
        },
        {
            "name": "Khong vuong",
            "A": [[1.0, 2.0, 3.0]],
            "should_raise": ValueError,
        },
    ]

    for case in test_cases:
        print(f"  - {case['name']}")
        try:
            d = determinant(case["A"])
            if case.get("should_raise"):
                assert False, "expected ValueError"
            assert abs(d - case["expected"]) < 1e-7, f"got {d}, want {case['expected']}"
            print("    => PASSED")
        except ValueError as err:
            if case.get("should_raise") == ValueError:
                print(f"    => PASSED (Caught expected error: {err})")
            else:
                print(f"    => FAILED: Unexpected error {err}")
                raise


if __name__ == "__main__":
    test_determinant()