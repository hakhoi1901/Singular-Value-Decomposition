"""
Thế ngược: giải Ux = c với U vuông tam giác trên (nghiệm duy nhất khi U khả nghịch).

Nếu gặp phần tử đường chéo ~0 (không thể thế ngược một cách duy nhất):
in "Hệ không có nghiệm duy nhất" và trả về [].
"""
from __future__ import annotations

import sys
from pathlib import Path
try:
    from part1.test_cases import BACK_SUBSTITUTION_TEST_CASES
except ImportError:
    from test_cases import BACK_SUBSTITUTION_TEST_CASES

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import is_zero, zero_rectify, TestLogger


def back_substitution(U: list[list[float]], c: list[float]) -> list[float]:
    """
    Giải hệ phương trình tam giác trên Ux = c từ dưới lên.

    Thuật toán:
        1. Kiểm tra trường hợp vô nghiệm (dòng toàn 0 nhưng vế phải khác 0).
        2. Nếu có biến tự do (đường chéo bằng 0), hệ có vô số nghiệm:
           - Tìm một nghiệm riêng x_0 bằng cách cho tất cả biến tự do = 0.
           - Tìm các vector cơ sở của không gian nghiệm (Null space basis).
           - In ra công thức nghiệm tổng quát.
        3. Nếu không có biến tự do, hệ có nghiệm duy nhất:
           - Tính nghiệm bằng phương pháp thế ngược.

    Tham số:
    - U: Ma trận tam giác trên vuông kích thước n × n
    - c: Vector vế phải kích thước n

    Trả về:
    - Nghiệm duy nhất nếu U khả nghịch
    - [] nếu U suy biến (vô nghiệm hoặc vô số nghiệm)

    Xử lý ngoại lệ:
    - Khi U[i][i] xấp xỉ 0: in cảnh báo và trả về [] (không dùng raise).
    """
    if not U:
        return []

    # Kiểm tra kích thước ma trận
    n = len(U)
    if any(len(row) != n for row in U):
        raise ValueError("Matrix U must be square (n x n)")
    if len(c) != n:
        raise ValueError("Vector c must have the same length as U")

    # Khởi tạo nghiệm
    x = [0.0] * n

    # Tìm các biến tự do
    free_vars = [i for i in range(n) if is_zero(U[i][i])]

    # Kiểm tra trường hợp vô nghiệm (dòng toàn 0 nhưng vế phải khác 0)
    for i in range(n):
        if all(is_zero(U[i][j]) for j in range(n)) and not is_zero(c[i]):
            print("Hệ vô nghiệm (No solution).")
            return []

    if free_vars:
        # Hệ có vô số nghiệm
        # Tìm nghiệm riêng x_0 (bằng cách cho tất cả biến tự do = 0)
        x_0 = [0.0] * n
        for i in range(n - 1, -1, -1):
            if i not in free_vars:
                s = c[i] - sum(U[i][j] * x_0[j] for j in range(i + 1, n))
                x_0[i] = zero_rectify(s / U[i][i])

        # Tìm các vector cơ sở của không gian nghiệm (Null space basis)
        null_basis = []
        for free_idx in free_vars:
            v = [0.0] * n
            v[free_idx] = 1.0  # Đặt tham số tự do này = 1, các biến tự do khác = 0
            for i in range(n - 1, -1, -1):
                if i not in free_vars:
                    s = -sum(U[i][j] * v[j] for j in range(i + 1, n))
                    v[i] = zero_rectify(s / U[i][i])
            null_basis.append(v)

        # In ra công thức nghiệm tổng quát
        print("\n[+] Hệ có vô số nghiệm. Công thức nghiệm tổng quát:")
        print(f"    x = {x_0}")
        for i, v in enumerate(null_basis):
            print(f"      + t_{i+1} * {v}")
        print("    (với t_i là các tham số tự do thuộc R)\n")
        
        return [] # Vẫn trả về rỗng để pass các test case expect_non_unique

    # Trường hợp nghiệm duy nhất
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = c[i] - sum(U[i][j] * x[j] for j in range(i + 1, n))
        x[i] = zero_rectify(s / U[i][i])

    return x


def test_back_substitution(test_cases: list[dict]):
    import os
    import sys
    import io                                 # Thêm dòng này
    from contextlib import redirect_stdout    # Thêm dòng này

    _parent = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if _parent not in sys.path:
        sys.path.insert(0, _parent) # Nên dùng insert(0, ...) thay vì insert(...)

    from verify_solution import verify_solution

    TestLogger.print_suite_header("Thế Ngược (Back Substitution)")
    
    passed_count = 0
    total_count = len(test_cases)

    for case in test_cases:
        try:
            trap = io.StringIO()
            with redirect_stdout(trap):
                x = back_substitution(case["U"], case["c"])

            if case.get("should_raise"):
                TestLogger.print_result(case['name'], False, "Lẽ ra phải phát sinh ValueError")
                continue
            
            if "expect_x" in case:
                assert len(x) == len(case["expect_x"])
                for i, v in enumerate(case["expect_x"]):
                    assert abs(x[i] - v) < 1e-6, f"x[{i}]={x[i]}"
            
            n = len(case["U"])
            if n > 0 and len(case["c"]) == n and x:
                e = verify_solution(case["U"], x, case["c"])
                assert e < 1e-8, f"||c-Ux||={e}"
                TestLogger.print_result(case['name'], True, f"(e = {e:.2e})")
                passed_count += 1
            else:
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
    test_back_substitution(BACK_SUBSTITUTION_TEST_CASES)
