import math
from config import zero_rectify


def verify_solution(A: list[list[float]], x: list[float], b: list[float]) -> float:
    n_rows = len(A)
    n_cols = len(A[0])

    # 1. Tính tích Ax (Ma trận A nhân vector x)
    # Kết quả Ax là một vector cột có n_rows phần tử
    Ax = []
    for i in range(n_rows):
        row_sum = 0.0
        for j in range(n_cols):
            row_sum += A[i][j] * x[j]
        Ax.append(row_sum)

    # 2. Tính vector dư r = b - Ax
    residual_vector = []
    for i in range(n_rows):
        diff = b[i] - Ax[i]
        residual_vector.append(diff)

    # 3. Tính chuẩn Euclidean (L2 Norm) của vector dư: e = sqrt(sum(ri^2))
    sum_squares = sum(ri ** 2 for ri in residual_vector)
    e = math.sqrt(sum_squares)

    # Sử dụng zero_rectify để xử lý nếu sai số quá nhỏ (xấp xỉ 0)
    return zero_rectify(e)

def test_verify_solution():
    test_cases = [
        {
            "name": "Nghiệm chính xác tuyệt đối",
            "A": [[1, 0], [0, 1]], "x": [1, 2], "b": [1, 2],
            "expected_e": 0.0
        },
        {
            "name": "Hệ 2x2 thông thường",
            "A": [[1, 2], [3, 4]], "x": [-2, 1], "b": [0, -2],
            "expected_e": 0.0
        },
        {
            "name": "Nghiệm có sai số nhỏ (do làm tròn)",
            "A": [[3]], "x": [0.333333333333333], "b": [1],
            "is_near_zero": True # e sẽ cực nhỏ nhưng có thể không bằng 0 tuyệt đối
        },
        {
            "name": "Nghiệm sai hoàn toàn",
            "A": [[1, 1], [1, 0]], "x": [1, 1], "b": [10, 10],
            "should_be_large": True
        },
        {
            "name": "Ma trận không, b không",
            "A": [[0, 0], [0, 0]], "x": [1, 5], "b": [0, 0],
            "expected_e": 0.0
        },
        {
            "name": "Ma trận 1x3 (Hệ thiếu phương trình)",
            "A": [[1, 2, 3]], "x": [1, 1, 1], "b": [6],
            "expected_e": 0.0
        }
    ]

    for case in test_cases:
        e = verify_solution(case['A'], case['x'], case['b'])
        print(f"Test {case['name']}: e = {e}")
        # Logic kiểm tra
        if "expected_e" in case:
            assert abs(e - case['expected_e']) < 1e-12
        elif case.get("is_near_zero"):
            assert e < 1e-10
        elif case.get("should_be_large"):
            assert e > 1.0
