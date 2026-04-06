import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from config import EPSILON, is_zero, zero_rectify

from config import EPSILON, is_zero, zero_rectify

def inverse(A: list[list[float]]) -> list[list[float]]:
    n = len(A)
    # Kiểm tra ma trận vuông
    if any(len(row) != n for row in A):
        raise ValueError("Matrix must be square")

    # Tạo ma trận bổ sung [A | I]
    # Sử dụng list comprehension để copy giá trị, tránh side effect
    matrix = [row + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(A)]

    # BƯỚC 1: Khử Gauss (Biến đổi về ma trận tam giác trên)
    for i in range(n):
        # Tìm phần tử chốt (pivoting) để đảm bảo ổn định số học
        pivot_row = i
        while pivot_row < n and is_zero(matrix[pivot_row][i]):
            pivot_row += 1

        if pivot_row == n:
            raise ValueError("Matrix is singular")

        # Đổi chỗ dòng hiện tại với dòng chứa phần tử chốt
        matrix[i], matrix[pivot_row] = matrix[pivot_row], matrix[i]

        # Khử các dòng phía dưới dòng i
        for j in range(i + 1, n):
            factor = matrix[j][i] / matrix[i][i]
            for k in range(i, 2 * n):
                matrix[j][k] -= factor * matrix[i][k]
            matrix[j][i] = 0.0

    # BƯỚC 2: Khử ngược (Biến đổi từ dưới lên để tạo ma trận đường chéo)
    for i in range(n - 1, -1, -1):
        for j in range(i - 1, -1, -1):
            factor = matrix[j][i] / matrix[i][i]
            for k in range(i, 2 * n):
                matrix[j][k] -= factor * matrix[i][k]
            matrix[j][i] = 0.0

    # BƯỚC 3: Chuẩn hóa đường chéo về 1 và trích xuất ma trận nghịch đảo
    inv_matrix = []
    for i in range(n):
        divisor = matrix[i][i]
        # Tạo dòng mới cho ma trận nghịch đảo (n cột cuối của matrix)
        new_row = []
        for j in range(n, 2 * n):
            val = matrix[i][j] / divisor
            new_row.append(zero_rectify(val))
        inv_matrix.append(new_row)

    return inv_matrix

def test_inverse():
    test_data = [
        {
            "name": "Ma trận đơn vị 2x2",
            "input": [[1.0, 0.0], [0.0, 1.0]],
            "expected": [[1.0, 0.0], [0.0, 1.0]]
        },
        {
            "name": "Ma trận số âm",
            "input": [[-1.0, -2.0], [-3.0, -4.0]],
            "expected": [[2.0, -1.0], [-1.5, 0.5]]
        },
        {
            "name": "Ma trận suy biến (Hàng 2 gấp đôi hàng 1)",
            "input": [[1.0, 2.0], [2.0, 4.0]],
            "should_raise": ValueError
        },
        {
            "name": "Ma trận 3x3",
            "input": [[1.0, 0.0, 5.0], [2.0, 1.0, 6.0], [3.0, 4.0, 0.0]],
            "expected": [[-24.0, 20.0, -5.0], [18.0, -15.0, 4.0], [5.0, -4.0, 1.0]]
        },
        {
            "name": "Ma trận có phần tử chốt bằng 0 ở giữa",
            "input": [[0.0, 1.0], [1.0, 0.0]],
            "expected": [[0.0, 1.0], [1.0, 0.0]]
        },
        {
            "name": "Ma trận 1x1",
            "input": [[5.0]],
            "expected": [[0.2]]
        },
        {
            "name": "Ma trận số thực nhỏ (gần suy biến)",
            "input": [[1.0, 2.0], [1.0, 2.000000000000001]], # Chênh lệch > EPSILON
            "expected": [[2000000000000001.0, -2000000000000000.0], [-1000000000000000.0, 1000000000000000.0]]
        },
        {
            "name": "Ma trận suy biến do số cực nhỏ (< EPSILON)",
            "input": [[1.0, 2.0], [1.0, 2.0 + 1e-15]],
            "should_raise": ValueError
        },
        {
            "name": "Ma trận không 3x3",
            "input": [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0]
            ],
            "should_raise": ValueError  # Kỳ vọng phải báo lỗi "Matrix is singular"
        }
    ]

    for case in test_data:
        print(f"Testing: {case['name']}")
        try:
            res = inverse(case['input'])
            # Kiểm tra kết quả (so sánh số thực dùng EPSILON)
            for i in range(len(res)):
                for j in range(len(res[0])):
                    assert abs(res[i][j] - case['expected'][i][j]) < 1e-9
            print("=> PASSED")
        except ValueError as e:
            if case.get("should_raise") == ValueError:
                print(f"=> PASSED (Caught expected error: {e})")
            else:
                print(f"=> FAILED: Unexpected error {e}")
