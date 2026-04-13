import sys
import os
import math

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from config import EPSILON, is_zero, zero_rectify, TestLogger
from test_cases import INVERSE_TEST_CASES
from utils import matmul, identity_matrix, check_identity

def inverse(A: list[list[float]]) -> list[list[float]]:
    """
    Tính ma trận nghịch đảo bằng phương pháp Gauss-Jordan.
    Phương pháp này sử dụng partial pivoting để tăng tính ổn định của thuật toán.
    
    Thuật toán:
        1. Tạo ma trận bổ sung [A | I]
        2. Khử Gauss (Biến đổi về ma trận tam giác trên có partial pivoting)
        3. Khử ngược (Tạo zeros ở nửa trên đường chéo)
        4. Chuẩn hóa đường chéo về 1 và trích xuất A^-1

    Tham số:
    - A: Ma trận vuông kích thước n × n

    Trả về:
    - Ma trận nghịch đảo của A

    Xử lý ngoại lệ:
    - Khi A không vuông: raise ValueError
    - Khi A suy biến: raise ValueError
    """
    n = len(A)
    if any(len(row) != n for row in A):
        raise ValueError("Matrix must be square")

    # Tạo ma trận bổ sung [A | I]
    im = identity_matrix(n)
    matrix = [row + im[i] for i, row in enumerate(A)]

    # BƯỚC 1: Khử Gauss (Biến đổi về ma trận tam giác trên có partial pivoting)
    for i in range(n):
        pivot_row = i
        # Tìm phần tử có trị tuyệt đối lớn nhất để làm chốt (Partial Pivoting)
        for r in range(i + 1, n):
            if abs(matrix[r][i]) > abs(matrix[pivot_row][i]):
                pivot_row = r

        if is_zero(matrix[pivot_row][i]):
            raise ValueError("Matrix is singular")

        # Hoán đổi dòng
        if pivot_row != i:
            matrix[i], matrix[pivot_row] = matrix[pivot_row], matrix[i]

        # Khử các phần tử dưới chốt
        for j in range(i + 1, n):
            factor = matrix[j][i] / matrix[i][i]
            for k in range(i, 2 * n):
                matrix[j][k] -= factor * matrix[i][k]
            matrix[j][i] = 0.0

    # BƯỚC 2: Khử ngược (Tạo zeros ở nửa trên đường chéo)
    for i in range(n - 1, -1, -1):
        for j in range(i - 1, -1, -1):
            factor = matrix[j][i] / matrix[i][i]
            for k in range(i, 2 * n):
                matrix[j][k] -= factor * matrix[i][k]
            matrix[j][i] = 0.0

    # BƯỚC 3: Chuẩn hóa đường chéo về 1 và trích xuất A^-1
    inv_matrix = []
    for i in range(n):
        divisor = matrix[i][i]
        new_row = []
        for j in range(n, 2 * n):
            val = matrix[i][j] / divisor
            new_row.append(zero_rectify(val))
        inv_matrix.append(new_row)

    return inv_matrix

# TESTING


def test_inverse(test_cases: list[dict]):
    TestLogger.print_suite_header("Tìm ma trận nghịch đảo A^-1")

    passed_tests = 0
    total_tests = len(test_cases)

    for case in test_cases:
        try:
            A = case['input']
            A_inv = inverse(A)
            
            # KIỂM CHỨNG TÍNH ĐÚNG ĐẮN: A * A^-1 == I
            I_approx = matmul(A, A_inv)
            
            if check_identity(I_approx):
                TestLogger.print_result(case['name'], True, "(A * A^-1 = I)")
                passed_tests += 1
            else:
                TestLogger.print_result(case['name'], False, "A * A^-1 không bằng I")

        except ValueError as e:
            if case.get("should_raise") == ValueError:
                TestLogger.print_result(case['name'], True, f"(Bắt đúng ngoại lệ: {e})")
                passed_tests += 1
            else:
                TestLogger.print_result(case['name'], False, f"(Lỗi không mong muốn: {e})")
        except Exception as e:
            TestLogger.print_result(case['name'], False, f"(Lỗi ngoài mong đợi: {e})")
                
    TestLogger.print_summary(passed_tests, total_tests)

if __name__ == "__main__":
    test_inverse(INVERSE_TEST_CASES)
