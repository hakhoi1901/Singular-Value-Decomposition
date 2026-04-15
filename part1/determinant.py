"""
Định thức: chỉ xét A vuông n × n

det(A) sau khi khử Gauss (partial pivot) đưa A về tam giác trên,
không bỏ qua cột — nếu không có pivot tại cột k thì det = 0.
"""
from __future__ import annotations

import sys
import io
from contextlib import redirect_stdout
from pathlib import Path
try:
    from part1.test_cases import DETERMINANT_TEST_CASES
except ImportError:
    from test_cases import DETERMINANT_TEST_CASES

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import zero_rectify, TestLogger
from gaussian import gaussian_eliminate


def determinant(A: list[list[float]]) -> float:
    """
    Tính det(A) với A ∈ R^(n×n) bằng khử Gauss có chọn phần tử chốt.
    Sử dụng lại hàm gaussian_eliminate đã được test kỹ lưỡng.

    Thuật toán:
        1. Kiểm tra A có vuông không. Nếu không, raise ValueError.
        2. Tạo một vector b giả (toàn số 0) có cùng số hàng với A.
        3. Gọi gaussian_eliminate(A, b_dummy) để đưa hệ về dạng tam giác trên.
           Hàm này trả về U (ma trận tam giác trên), _ (không dùng), và swap_count (số lần hoán vị).
        4. Tính tích các phần tử trên đường chéo chính của U: det_a = U[0][0] * U[1][1] * ... * U[n-1][n-1].
        5. Nếu số lần hoán vị (swap_count) là lẻ, nhân det_a với -1.
        6. Trả về det_a (đã được làm tròn zero_rectify).

    Tham số:
    - A: Ma trận vuông kích thước n × n

    Trả về:
    - Định thức của A

    Xử lý ngoại lệ:
    - Khi A không vuông: raise ValueError
    """
    if not A:
        return 1.0

    n = len(A)
    if any(len(row) != n for row in A):
        raise ValueError("A phải vuông (n × n) để tính định thức")

    b_dummy = [0.0] * n
    
    trap = io.StringIO()
    with redirect_stdout(trap):
        U, _, swap_count = gaussian_eliminate(A, b_dummy)

    det_a = 1.0
    for i in range(n):
        det_a *= U[i][i]
        
    if swap_count % 2 != 0:
        det_a = -det_a
        
    return zero_rectify(det_a)


def test_determinant(test_cases: list[dict]):
    import warnings
    warnings.simplefilter("ignore", UserWarning) # Bỏ qua warning pivot nhỏ
    
    TestLogger.print_suite_header("Định Thức (Determinant)")
    
    passed_count = 0
    total_count = len(test_cases)

    for case in test_cases:
        try:
            d = determinant(case["A"])
            if case.get("should_raise"):
                TestLogger.print_result(case['name'], False, "Lẽ ra phải phát sinh ValueError")
                continue
            assert abs(d - case["expected"]) < 1e-7, f"got {d}, want {case['expected']}"
            TestLogger.print_result(case['name'], True)
            passed_count += 1
        except ValueError as err:
            if case.get("should_raise") == ValueError:
                TestLogger.print_result(case['name'], True, f"(Bắt đúng lỗi: {err})")
                passed_count += 1
            else:
                TestLogger.print_result(case['name'], False, f"(Lỗi ngoài mong đợi: {err})")
        except AssertionError as err:
            TestLogger.print_result(case['name'], False, f"(Assertion: {err})")
            
    TestLogger.print_summary(passed_count, total_count)


if __name__ == "__main__":
    test_determinant(DETERMINANT_TEST_CASES)