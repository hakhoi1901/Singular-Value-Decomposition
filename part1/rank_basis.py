import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from config import is_zero, zero_rectify


def to_rref(A: list[list[float]]):
    n_rows = len(A)
    n_cols = len(A[0])
    rref = [row[:] for row in A]  # Copy ma trận
    pivot_row = 0
    pivot_cols = []

    for j in range(n_cols):
        if pivot_row >= n_rows:
            break

        # Tìm dòng chốt
        sel_row = pivot_row
        while sel_row < n_rows and is_zero(rref[sel_row][j]):
            sel_row += 1

        if sel_row == n_rows:
            continue  # Cột này toàn số 0 ở phía dưới, nhảy sang cột kế

        pivot_cols.append(j)
        rref[pivot_row], rref[sel_row] = rref[sel_row], rref[pivot_row]

        # Chuẩn hóa hàng chốt về 1
        divisor = rref[pivot_row][j]
        rref[pivot_row] = [zero_rectify(x / divisor) for x in rref[pivot_row]]

        # Khử các dòng khác về 0 (cả trên và dưới)
        for i in range(n_rows):
            if i != pivot_row:
                factor = rref[i][j]
                for k in range(j, n_cols):
                    rref[i][k] = zero_rectify(rref[i][k] - factor * rref[pivot_row][k])

        pivot_row += 1

    return rref, pivot_cols


def rank_and_basis(A: list[list[float]]):
    if not A or not A[0]:
        return 0, [], [], []

    n_rows = len(A)
    n_cols = len(A[0])
    rref, pivot_cols = to_rref(A)

    # 1. Rank
    rank = len(pivot_cols)

    # 2. Row Basis (Các dòng khác 0 trong RREF)
    row_basis = [row for row in rref if any(not is_zero(x) for x in row)]

    # 3. Column Basis (Cột tại vị trí pivot lấy từ ma trận gốc A)
    # Trả về dạng danh sách các vector cột
    col_basis = []
    for j in pivot_cols:
        col_vector = [A[i][j] for i in range(n_rows)]
        col_basis.append(col_vector)

    # 4. Null Basis (Cơ sở không gian nghiệm Ax = 0)
    null_basis = []
    free_cols = [j for j in range(n_cols) if j not in pivot_cols]

    for free_j in free_cols:
        # Tạo vector nghiệm đặc biệt bằng cách gán biến tự do = 1, các biến tự do khác = 0
        special_solution = [0.0] * n_cols
        special_solution[free_j] = 1.0

        for i, p_col in enumerate(pivot_cols):
            # Từ RREF: x_pivot + sum(rref[i][free_j] * x_free) = 0
            # => x_pivot = -rref[i][free_j]
            special_solution[p_col] = zero_rectify(-rref[i][free_j])

        null_basis.append(special_solution)

    return rank, col_basis, row_basis, null_basis

def compare_matrices(M1, M2):
    if len(M1) != len(M2) or (len(M1) > 0 and len(M1[0]) != len(M2[0])):
        return False
    for i in range(len(M1)):
        for j in range(len(M1[0])):
            if not is_zero(M1[i][j] - M2[i][j]):
                return False
    return True


def test_rank_and_basis():
    test_cases = [
        {
            "name": "Ma trận đơn vị 3x3 (Full Rank)",
            "input": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "exp_rank": 3,
            "null_is_empty": True
        },
        {
            "name": "Ma trận không 2x3",
            "input": [[0, 0, 0], [0, 0, 0]],
            "exp_rank": 0,
            "exp_null_dim": 3
        },
        {
            "name": "Ma trận có dòng phụ thuộc tuyến tính (R3 = R1 + R2)",
            "input": [[1, 2, 1], [0, 1, 1], [1, 3, 2]],
            "exp_rank": 2,
            "exp_null_dim": 1
        },
        {
            "name": "Ma trận 1 dòng nhiều cột",
            "input": [[1, 2, 3, 4]],
            "exp_rank": 1,
            "exp_null_dim": 3
        },
        {
            "name": "Ma trận có cột toàn số 0 ở giữa",
            "input": [[1, 0, 2], [3, 0, 4]],
            "exp_rank": 2,
            "exp_null_dim": 1
        },
        {
            "name": "Ma trận vuông suy biến",
            "input": [[1, 1], [1, 1]],
            "exp_rank": 1,
            "exp_null_dim": 1
        },
        {
            "name": "Ma trận số thực cực nhỏ (< EPSILON)",
            "input": [[1, 2], [0, 1e-16]],
            "exp_rank": 1,
            "exp_null_dim": 1
        },
        {
            "name": "Ma trận kích thước 1x1",
            "input": [[5.0]],
            "exp_rank": 1,
            "exp_null_dim": 0
        }
    ]

    for case in test_cases:
        print(f"  - {case['name']}")
        rank, col_b, row_b, null_b = rank_and_basis(case['input'])

        # Kiểm tra Hạng
        assert rank == case['exp_rank'], f"Sai Rank: kỳ vọng {case['exp_rank']}, thực tế {rank}"

        # Kiểm tra số lượng vector cơ sở
        assert len(col_b) == rank, "Số lượng Column Basis phải bằng Rank"
        assert len(row_b) == rank, "Số lượng Row Basis phải bằng Rank"

        if case.get("exp_null_dim") is not None:
            assert len(null_b) == case['exp_null_dim'], f"Sai Null Basis dim: kỳ vọng {case['exp_null_dim']}"

        print("    => PASSED")

if __name__ == "__main__":
    test_rank_and_basis()
