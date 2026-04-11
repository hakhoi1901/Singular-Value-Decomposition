"""
Khử Gauss với partial pivoting cho hệ Ax = b với A ∈ R^(m×n).

Vô số nghiệm (khi rank(A) < n và hệ tương thích): nghiệm tổng quát có dạng
    x = x_0 + t_1 * v_1 + ... + t_{n-r} * v_{n-r}
trong đó x_0 là một nghiệm riêng, v_k là cơ sở không gian nghiệm N(A), r = rank(A),
t_k ∈ R là tham số tự do.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import is_zero, zero_rectify, TestLogger
from back_substitution import back_substitution
from test_cases import GAUSSIAN_ELIMINATE_TEST_CASES
from utils import is_upper_triangular


def gaussian_eliminate(A: list[list[float]], b: list[float]):
    """
    Giải hệ phương trình tuyến tính Ax = b bằng phương pháp khử Gauss.

    Thuật toán:
        1. Đưa [A | b] về dạng bậc thang dòng (REF) bằng khử Gauss có chọn phần tử chốt cột.
        2. A có kích thước m × n (m hàng, n cột). len(b) phải bằng m.
        3. Khi không tìm được pivot tại một cột (các phần tử dưới đều xấp xỉ 0): in
           "Không có pivot tại cột k" và bỏ qua cột đó (không raise).
        4. Nếu sau đó gọi "back_substitution" trên tam giác thu được mà gặp đường chéo ~0,
           hàm đó sẽ in "Hệ không có nghiệm duy nhất" và trả về "[]".

    Tham số:
        A: Ma trận hệ số (m × n)
        b: Vector vế phải (m,)

    Trả về:
        U: ma trận hệ số sau biến đổi (m × n), dạng bậc thang dòng
        x: vector nghiệm (độ dài n), trả về [] nếu vô số nghiệm hoặc vô nghiệm
        swap_count: số lần hoán đổi dòng

    Xử lý ngoại lệ:
        - Khi A rỗng: raise ValueError
        - Khi A không vuông: raise ValueError
        - Khi len(b) != m: raise ValueError
    """

    # Kiểm tra ma trận
    if not A:
        raise ValueError("Ma trận A rỗng")

    # Kiểm tra kích thước ma trận
    m = len(A)
    n = len(A[0])
    if any(len(row) != n for row in A):
        raise ValueError("Mọi hàng của A phải cùng số cột")

    # Kiểm tra vector b
    if len(b) != m:
        raise ValueError("Vector b phải có độ dài bằng số hàng của A")

    # Tạo ma trận mở rộng
    M = [list(A[i]) + [float(b[i])] for i in range(m)]
    swap_count = 0
    pivot_row = 0

    # Vòng lặp chính
    for j in range(n):
        if pivot_row >= m:
            break

        # Tìm phần tử chốt
        p = pivot_row
        best = abs(M[pivot_row][j])
        for i in range(pivot_row + 1, m):
            v = abs(M[i][j])
            if v > best:
                best = v
                p = i
        
        # Kiểm tra pivot
        pivot = M[p][j]

        # Nếu pivot bằng 0, bỏ qua cột
        if is_zero(pivot):
            print(f"Không có pivot tại cột {j + 1}")
            continue

        # Kiểm tra pivot nhỏ
        if abs(pivot) < 1e-8:
            warnings.warn(
                "Pivot is very small; the system may be ill-conditioned.",
                UserWarning,
                stacklevel=2,
            )

        if p != pivot_row:
            M[pivot_row], M[p] = M[p], M[pivot_row]
            swap_count += 1

        pivot = M[pivot_row][j]
        for i in range(pivot_row + 1, m):
            if is_zero(M[i][j]):
                continue
            lik = M[i][j] / pivot
            for k in range(j, n + 1):
                M[i][k] -= lik * M[pivot_row][k]
                M[i][k] = zero_rectify(M[i][k])

        pivot_row += 1

    U = [row[:n] for row in M]
    c = [row[n] for row in M]
    
    if m == n:
        x = back_substitution(U, c)
    else:
        x = []

    return U, x, swap_count




def test_gaussian_eliminate(test_cases: list[dict]):
    import os
    import sys
    import warnings

    _parent = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if _parent not in sys.path:
        sys.path.insert(0, _parent)

    from verify_solution import verify_solution

    warnings.simplefilter("ignore", UserWarning)

    TestLogger.print_suite_header("Khử Gauss (Gaussian eliminate)")
    
    passed_count = 0
    total_count = len(test_cases)

    for case in test_cases:
        try:
            U, x, swaps = gaussian_eliminate(case["A"], case["b"])
            
            if case.get("should_raise"):
                TestLogger.print_result(case['name'], False, "Lẽ ra phải phát sinh ValueError")
                continue
                
            if not case.get("skip_backsub"):
                assert is_upper_triangular(U), "U phai tam giac tren (he vuong day du hang)"
                
            if "expect_swaps" in case:
                assert swaps == case["expect_swaps"], f"swap_count={swaps}"
                
            if case.get("skip_backsub"):
                TestLogger.print_result(case['name'], True)
                passed_count += 1
                continue
                
            if case.get("expect_non_unique"):
                assert x == [], "Expected empty solution for non-unique system"
                TestLogger.print_result(case['name'], True)
                passed_count += 1
                continue
                
            if "expect_x" in case:
                for i, v in enumerate(case["expect_x"]):
                    assert abs(x[i] - v) < 1e-8, f"x[{i}]={x[i]}"
                    
            e = verify_solution(case["A"], x, case["b"])
            assert e < 1e-8, f"||b-Ax||={e}"
            TestLogger.print_result(case['name'], True, f"(e = {e:.2e})")
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
    test_gaussian_eliminate(GAUSSIAN_ELIMINATE_TEST_CASES)
