import sys
import os
import math

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from config import EPSILON, is_zero, zero_rectify

def inverse(A: list[list[float]]) -> list[list[float]]:
    """
    Tính ma trận nghịch đảo bằng phương pháp Gauss-Jordan.
    """
    n = len(A)
    if any(len(row) != n for row in A):
        raise ValueError("Matrix must be square")

    # Tạo ma trận bổ sung [A | I]
    matrix = [row + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(A)]

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

def multiply_matrix(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    """Nhân hai ma trận vuông A và B."""
    n = len(A)
    C = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C

def check_identity(M: list[list[float]], tol: float = 1e-5) -> bool:
    """Kiểm tra ma trận M có xấp xỉ ma trận đơn vị I hay không."""
    n = len(M)
    for i in range(n):
        for j in range(n):
            expected = 1.0 if i == j else 0.0
            if abs(M[i][j] - expected) > tol:
                print(f"      [!] Lệch tại [{i}][{j}]: {M[i][j]} khác {expected}")
                return False
    return True

def test_inverse():
    print("\nKhởi chạy Test Suite: Tìm ma trận nghịch đảo A^-1\n" + "-"*50)
    
    test_data = [
        {
            "name": "Ma trận đơn vị 2x2",
            "input": [[1.0, 0.0], [0.0, 1.0]]
        },
        {
            "name": "Ma trận số âm",
            "input": [[-1.0, -2.0], [-3.0, -4.0]]
        },
        {
            "name": "Ma trận suy biến (Hàng 2 gấp đôi hàng 1)",
            "input": [[1.0, 2.0], [2.0, 4.0]],
            "should_raise": ValueError
        },
        {
            "name": "Ma trận 3x3",
            "input": [[1.0, 0.0, 5.0], [2.0, 1.0, 6.0], [3.0, 4.0, 0.0]]
        },
        {
            "name": "Ma trận có phần tử chốt bằng 0 ở giữa",
            "input": [[0.0, 1.0], [1.0, 0.0]]
        },
        {
            "name": "Ma trận số thực nhỏ (Điều kiện kém / Ill-conditioned)",
            "input": [[1.0, 2.0], [1.0, 2.00000000001]]
        },
        {
            "name": "Ma trận suy biến do số cực nhỏ (< EPSILON)",
            "input": [[1.0, 2.0], [1.0, 2.0 + 1e-16]],
            "should_raise": ValueError
        }
    ]

    passed_tests = 0
    total_tests = len(test_data)

    for case in test_data:
        print(f"  - {case['name']}")
        try:
            A = case['input']
            A_inv = inverse(A)
            
            # KIỂM CHỨNG TÍNH ĐÚNG ĐẮN: A * A^-1 == I
            I_approx = multiply_matrix(A, A_inv)
            
            if check_identity(I_approx):
                print("    => PASSED (A * A^-1 = I)")
                passed_tests += 1
            else:
                print("    => FAILED: A * A^-1 không bằng I")

        except ValueError as e:
            if case.get("should_raise") == ValueError:
                print(f"    => PASSED (Bắt đúng ngoại lệ: {e})")
                passed_tests += 1
            else:
                print(f"    => FAILED: Lỗi không mong muốn -> {e}")
                
    print(f"\nTổng kết: {passed_tests}/{total_tests} tests passed.\n")

if __name__ == "__main__":
    test_inverse()