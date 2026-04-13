import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from config import is_zero, zero_rectify, TestLogger


def to_rref(A: list[list[float]], tol: float = None):
    """
    Đưa ma trận A về dạng bậc thang rút gọn (RREF) bằng khử Gauss-Jordan.

    Thuật toán:
        1. Duyệt qua từng cột của ma trận.
        2. Tìm dòng có phần tử khác 0 lớn nhất (pivot) tại cột hiện tại (partial pivoting).
        3. Hoán đổi dòng để đưa pivot lên vị trí hiện tại.
        4. Chuẩn hóa dòng pivot để phần tử chốt bằng 1.
        5. Khử các phần tử khác 0 trong cột hiện tại (cả trên và dưới dòng pivot).
        6. Lặp lại cho đến khi toàn bộ ma trận được đưa về RREF.

    Tham số:
        A: Ma trận đầu vào (m × n)

    Trả về:
        - rref: Ma trận sau khi đưa về RREF
        - pivot_cols: Danh sách các chỉ số cột chứa pivot
    """
    n_rows = len(A)
    n_cols = len(A[0])
    rref = [row[:] for row in A]  # Copy ma trận
    pivot_row = 0
    pivot_cols = []

    for j in range(n_cols):
        if pivot_row >= n_rows:
            break

        # Tìm dòng chốt với partial pivoting
        sel_row = pivot_row
        best_val = abs(rref[pivot_row][j])
        for r in range(pivot_row + 1, n_rows):
            val = abs(rref[r][j])
            if val > best_val:
                best_val = val
                sel_row = r

        is_pivot_zero = (abs(rref[sel_row][j]) <= tol) if tol is not None else is_zero(rref[sel_row][j])
        if is_pivot_zero:
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


def rank_and_basis(A: list[list[float]], tol: float = None):
    """
    Tính hạng và tìm cơ sở cho không gian cột, không gian hàng và không gian null của ma trận A.

    Thuật toán:
        1. Đưa ma trận A về dạng bậc thang rút gọn (RREF) bằng hàm to_rref().
        2. Hạng (rank) của ma trận bằng số lượng pivot (số cột chứa pivot).
        3. Cơ sở không gian cột (Column Basis): Các cột tương ứng của ma trận gốc A tại vị trí pivot.
        4. Cơ sở không gian hàng (Row Basis): Các dòng khác 0 trong ma trận RREF.
        5. Cơ sở không gian null (Null Basis): Tìm nghiệm của hệ phương trình tuyến tính Ax = 0
           bằng cách sử dụng ma trận RREF.

    Tham số:
        A: Ma trận đầu vào (m × n)

    Trả về:
        - rank: Hạng của ma trận
        - col_basis: Danh sách các vector cơ sở không gian cột
        - row_basis: Danh sách các vector cơ sở không gian hàng
        - null_basis: Danh sách các vector cơ sở không gian null

    Xử lý ngoại lệ:
        - Nếu A là ma trận rỗng, trả về 0, [], [], []
    """
    if not A or not A[0]:
        return 0, [], [], []

    n_rows = len(A)
    n_cols = len(A[0])
    rref, pivot_cols = to_rref(A, tol)

    # 1. Rank
    rank = len(pivot_cols)

    row_basis = [row for row in rref if any(not ((abs(x) <= tol) if tol is not None else is_zero(x)) for x in row)]

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


def test_rank_and_basis(test_cases: list[dict]):
    TestLogger.print_suite_header("TÌM HẠNG VÀ CƠ SỞ (RANK & BASIS)")
    
    passed_count = 0
    total_count = len(test_cases)

    for case in test_cases:
        try:
            rank, col_b, row_b, null_b = rank_and_basis(case['input'])

            # Kiểm tra Hạng
            assert rank == case['exp_rank'], f"Sai Rank: kỳ vọng {case['exp_rank']}, thực tế {rank}"

            # Kiểm tra số lượng vector cơ sở
            assert len(col_b) == rank, "Số lượng Column Basis phải bằng Rank"
            assert len(row_b) == rank, "Số lượng Row Basis phải bằng Rank"

            if case.get("exp_null_dim") is not None:
                assert len(null_b) == case['exp_null_dim'], f"Sai Null Basis dim: kỳ vọng {case['exp_null_dim']}"

            TestLogger.print_result(case['name'], True)
            passed_count += 1
        except AssertionError as e:
            TestLogger.print_result(case['name'], False, f"(Assertion: {e})")

    TestLogger.print_summary(passed_count, total_count)

if __name__ == "__main__":
    test_rank_and_basis(RANK_BASIS_TEST_CASES)
