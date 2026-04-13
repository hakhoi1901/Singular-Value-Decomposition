"""
Cài đặt phân rã SVD không dùng các hàm phân rã có sẵn.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import EPSILON, TestLogger, is_zero, zero_rectify
from test_cases import DECOMPOSITION_TEST_CASES
from utils import matmul, transpose, dot_product, vector_norm, identity_matrix, matvec, orthogonalize, find_new_unit_vector, rectify_matrix, rectify_vector, max_abs_diff

def _validate_matrix(A: list[list[float]]) -> None:
    """
    Kiểm tra tính hợp lệ của ma trận A.

    Tham số:
        A: Ma trận cần kiểm tra

    Trả về:
        None

    Xử lý ngoại lệ:
        ValueError: Nếu ma trận không hợp lệ
    """
    if not A:
        raise ValueError("Ma trận A không được rỗng")

    n_cols = len(A[0])
    if n_cols == 0:
        raise ValueError("Ma trận A phải có ít nhất 1 cột")

    for row in A:
        if len(row) != n_cols:
            raise ValueError("Tất cả các dòng của A phải cùng số cột")



def _jacobi_eigen_symmetric(S: list[list[float]]) -> tuple[list[float], list[list[float]]]:
    """
    Tính trị riêng và vector riêng của ma trận đối xứng S bằng phương pháp Jacobi.

    Tham số:
        S: Ma trận đối xứng cần tính trị riêng và vector riêng

    Trả về:
        Tuple (eigenvalues, eigenvectors) chứa danh sách trị riêng và ma trận vector riêng

    Xử lý ngoại lệ:
        ValueError: Nếu ma trận không phải là ma trận vuông
    """
    n = len(S)
    D = [row[:] for row in S]
    V = identity_matrix(n)

    if n == 1:
        return [float(D[0][0])], [[1.0]]

    max_iter = max(30, 20 * n * n)
    for _ in range(max_iter):
        p = 0
        q = 1
        max_off = abs(D[p][q])
        for i in range(n):
            for j in range(i + 1, n):
                value = abs(D[i][j])
                if value > max_off:
                    max_off = value
                    p, q = i, j

        if max_off < EPSILON:
            break
        if is_zero(D[p][q]):
            continue

        tau = (D[q][q] - D[p][p]) / (2.0 * D[p][q])
        sign_tau = 1.0 if tau >= 0.0 else -1.0
        t = sign_tau / (abs(tau) + math.sqrt(1.0 + tau * tau))
        c = 1.0 / math.sqrt(1.0 + t * t)
        s = t * c

        app = D[p][p]
        aqq = D[q][q]
        apq = D[p][q]

        D[p][p] = app - t * apq
        D[q][q] = aqq + t * apq
        D[p][q] = 0.0
        D[q][p] = 0.0

        for k in range(n):
            if k == p or k == q:
                continue
            dkp = D[k][p]
            dkq = D[k][q]
            D[k][p] = c * dkp - s * dkq
            D[p][k] = D[k][p]
            D[k][q] = s * dkp + c * dkq
            D[q][k] = D[k][q]

        for k in range(n):
            vkp = V[k][p]
            vkq = V[k][q]
            V[k][p] = c * vkp - s * vkq
            V[k][q] = s * vkp + c * vkq

    eigenvalues: list[float] = []
    for i in range(n):
        value = zero_rectify(D[i][i])
        if value < 0.0 and is_zero(value):
            value = 0.0
        eigenvalues.append(float(value))

    order = sorted(range(n), key=lambda idx: eigenvalues[idx], reverse=True)
    eigen_sorted = [eigenvalues[idx] for idx in order]

    V_sorted = [[0.0 for _ in range(n)] for _ in range(n)]
    for new_col, old_col in enumerate(order):
        for row in range(n):
            V_sorted[row][new_col] = float(V[row][old_col])

    cols: list[list[float]] = []
    for j in range(n):
        col = [V_sorted[i][j] for i in range(n)]
        col = orthogonalize(col, cols)
        col_norm = vector_norm(col)
        if is_zero(col_norm):
            col = find_new_unit_vector(cols, n)
        else:
            col = [zero_rectify(x / col_norm) for x in col]
        cols.append(col)

    V_orth = [[0.0 for _ in range(n)] for _ in range(n)]
    for j, col in enumerate(cols):
        for i, value in enumerate(col):
            V_orth[i][j] = value

    return eigen_sorted, V_orth


def decompose_svd(A: list[list[float]]) -> tuple[list[list[float]], list[float], list[list[float]]]:
    """
    Phân rã SVD cho ma trận A.

    Tham số:
        A: Ma trận cần phân rã SVD

    Trả về:
        (U, Sigma, V_T)
        - U là ma trận trực giao trái kích thước m x m
        - Sigma là danh sách singular values giảm dần (độ dài min(m, n))
        - V_T là chuyển vị của ma trận trực giao phải kích thước n x n
    """
    _validate_matrix(A)

    A_num = [[float(value) for value in row] for row in A]
    m = len(A_num)
    n = len(A_num[0])
    r = min(m, n)

    A_T = transpose(A_num)
    A_T_A = matmul(A_T, A_num)
    eigenvalues, V = _jacobi_eigen_symmetric(A_T_A)

    sigma_all: list[float] = []
    for lam in eigenvalues:
        if lam < 0.0:
            lam = 0.0 if is_zero(lam) else 0.0
        sigma_all.append(math.sqrt(lam) if lam > 0.0 else 0.0)

    Sigma = rectify_vector(sigma_all[:r])

    u_cols: list[list[float]] = []
    for i in range(r):
        sigma_i = Sigma[i]
        if not is_zero(sigma_i):
            v_i = [V[row][i] for row in range(n)]
            candidate = matvec(A_num, v_i)
            candidate = [value / sigma_i for value in candidate]
            candidate = orthogonalize(candidate, u_cols)
            norm_candidate = vector_norm(candidate)
            if not is_zero(norm_candidate):
                u_cols.append([zero_rectify(x / norm_candidate) for x in candidate])
                continue

        u_cols.append(find_new_unit_vector(u_cols, m))

    while len(u_cols) < m:
        u_cols.append(find_new_unit_vector(u_cols, m))

    U = [[u_cols[col][row] for col in range(m)] for row in range(m)]
    V_T = transpose(V)

    U = rectify_matrix(U)
    V_T = rectify_matrix(V_T)
    return U, Sigma, V_T


def _sigma_to_matrix(sigma: list[float], m: int, n: int) -> list[list[float]]:
    """
    Chuyển đổi danh sách singular values thành ma trận Sigma.

    Tham số:
        sigma: Danh sách singular values
        m: Số dòng của ma trận Sigma
        n: Số cột của ma trận Sigma

    Trả về:
        Ma trận Sigma
    """
    S = [[0.0 for _ in range(n)] for _ in range(m)]
    r = min(len(sigma), m, n)
    for i in range(r):
        S[i][i] = sigma[i]
    return S




def _is_descending_nonnegative(values: list[float], tol: float = 1e-9) -> bool:
    """
    Kiểm tra xem danh sách có giảm dần và không âm hay không.

    Tham số:
        values: Danh sách cần kiểm tra
        tol: Độ dung sai

    Trả về:
        True nếu danh sách giảm dần và không âm, False ngược lại
    """
    for v in values:
        if v < -tol:
            return False

    for i in range(len(values) - 1):
        if values[i] + tol < values[i + 1]:
            return False
    return True


def _is_orthogonal(Q: list[list[float]], tol: float = 1e-7) -> bool:
    """
    Kiểm tra xem ma trận có trực giao hay không.

    Tham số:
        Q: Ma trận cần kiểm tra
        tol: Độ dung sai

    Trả về:
        True nếu ma trận trực giao, False ngược lại
    """
    if not Q:
        return True

    n = len(Q)
    if any(len(row) != n for row in Q):
        return False

    QT = transpose(Q)
    QTQ = matmul(QT, Q)
    I = identity_matrix(n)
    err = max_abs_diff(QTQ, I)
    return err <= tol


def test_decompose_svd(test_cases: list[dict]) -> None:
    """
    Bộ kiểm thử cho decompose_svd theo nhiều dạng ma trận thường gặp.
    """
    TestLogger.print_suite_header("Phân Rã Giá Trị Suy Biến (SVD)")
    
    passed_count = 0
    total_count = len(test_cases)

    for case in test_cases:
        try:
            A = case["input"]
            U, sigma, V_T = decompose_svd(A)

            if case.get("should_raise") == ValueError:
                TestLogger.print_result(case['name'], False, "Lẽ ra phải phát sinh ValueError")
                continue

            m = len(A)
            n = len(A[0])
            r = min(m, n)

            assert len(U) == m and len(U[0]) == m, "U phải có kích thước m x m"
            assert len(sigma) == r, "Sigma phải có độ dài min(m, n)"
            assert len(V_T) == n and len(V_T[0]) == n, "V_T phải có kích thước n x n"

            assert _is_descending_nonnegative(sigma), "Sigma phải không âm và giảm dần"
            assert _is_orthogonal(U), "U phải là ma trận trực giao"

            V = transpose(V_T)
            assert _is_orthogonal(V), "V phải là ma trận trực giao"

            S = _sigma_to_matrix(sigma, m, n)
            A_reconstructed = matmul(matmul(U, S), V_T)
            A_float = [[float(value) for value in row] for row in A]

            err = max_abs_diff(A_reconstructed, A_float)
            tol = case.get("tol", 1e-7)
            assert err <= tol, f"Sai số tái tạo quá lớn: {err:.3e} > {tol}"

            # Nếu chạy đến đây là thành công
            TestLogger.print_result(case['name'], True, f"(err = {err:.2e})")
            passed_count += 1

        except ValueError as e:
            if case.get("should_raise") == ValueError:
                TestLogger.print_result(case['name'], True, f"(Bắt đúng lỗi: {e})")
                passed_count += 1
            else:
                TestLogger.print_result(case['name'], False, f"(Lỗi ngoài mong đợi: {e})")
        except AssertionError as e:
            TestLogger.print_result(case['name'], False, f"(Assertion: {e})")

    TestLogger.print_summary(passed_count, total_count)

if __name__ == "__main__":
    test_decompose_svd(DECOMPOSITION_TEST_CASES)
