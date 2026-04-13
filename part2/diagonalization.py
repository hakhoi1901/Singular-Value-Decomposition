"""
Cài đặt chéo hóa ma trận không dùng các hàm giải thuật có sẵn.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import EPSILON, is_zero, zero_rectify, TestLogger
try:
    from part2.test_cases import DIAGONALIZATION_TEST_CASES
except ImportError:
    from test_cases import DIAGONALIZATION_TEST_CASES
from part1.rank_basis import to_rref, rank_and_basis
from utils import matmul, transpose, dot_product, vector_norm, identity_matrix, normalize, is_zero_tol, rectify_matrix, rectify_vector, max_abs_diff, orthogonalize, find_new_unit_vector


def _validate_square_matrix(A: list[list[float]]) -> None:
    """
    Kiểm tra xem ma trận có phải là ma trận vuông hay không.

    Tham số:
        A: Ma trận cần kiểm tra

    Trả về:
        None

    Xử lý ngoại lệ:
        ValueError: Nếu A không phải là ma trận vuông
    """
    if not A:
        raise ValueError("Ma trận A không được rỗng")

    n_cols = len(A[0])
    if n_cols == 0:
        raise ValueError("Ma trận A phải có ít nhất 1 cột")

    for row in A:
        if len(row) != n_cols:
            raise ValueError("Tất cả các dòng của A phải cùng số cột")

    if len(A) != n_cols:
        raise ValueError("A phải là ma trận vuông")





def normalize(v: list[float]) -> list[float]:
    """
    Chuẩn hóa vector v về vector đơn vị.

    Tham số:
        v: Vector cần chuẩn hóa

    Trả về:
        Vector đã được chuẩn hóa
    """
    nv = vector_norm(v)
    if is_zero(nv):
        return [0.0 for _ in v]
    return [zero_rectify(x / nv) for x in v]


def _qr_decomposition(A: list[list[float]]) -> tuple[list[list[float]], list[list[float]]]:
    """
    Phân tích QR của ma trận A bằng thuật toán Gram-Schmidt.

    Tham số:
        A: Ma trận cần phân tích

    Trả về:
        Tuple (Q, R) trong đó Q là ma trận trực giao, R là ma trận tam giác trên
    """
    n = len(A)
    q_cols: list[list[float]] = []
    R = [[0.0 for _ in range(n)] for _ in range(n)]

    for j in range(n):
        v = [A[i][j] for i in range(n)]

        for i in range(j):
            q_i = q_cols[i]
            r_ij = dot_product(q_i, v)
            R[i][j] = r_ij
            if not is_zero(r_ij):
                for k in range(n):
                    v[k] = zero_rectify(v[k] - r_ij * q_i[k])

        r_jj = vector_norm(v)
        if is_zero(r_jj):
            q_j = find_new_unit_vector(q_cols, n)
            R[j][j] = 0.0
        else:
            q_j = [zero_rectify(x / r_jj) for x in v]
            R[j][j] = r_jj

        q_cols.append(q_j)

    Q = [[q_cols[col][row] for col in range(n)] for row in range(n)]
    Q = rectify_matrix(Q)
    R = rectify_matrix(R)
    return Q, R


def _qr_eigen_diagonal(A: list[list[float]], tol: float = 1e-10, max_iter: int = 4000) -> list[float]:
    """
    Tìm giá trị riêng của ma trận A bằng thuật toán QR.

    Tham số:
        A: Ma trận cần tìm giá trị riêng
        tol: Độ dung sai
        max_iter: Số lần lặp tối đa

    Trả về:
        Danh sách các giá trị riêng

    Xử lý ngoại lệ:
        ValueError: Nếu thuật toán không hội tụ
    """
    n = len(A)
    A_k = [row[:] for row in A]

    for _ in range(max_iter):
        Q, R = _qr_decomposition(A_k)
        A_k = matmul(R, Q)
        A_k = rectify_matrix(A_k)

        max_subdiag = 0.0
        for i in range(1, n):
            for j in range(i):
                value = abs(A_k[i][j])
                if value > max_subdiag:
                    max_subdiag = value

        if max_subdiag <= tol:
            return [float(A_k[i][i]) for i in range(n)]

    raise ValueError("Không hội tụ QR; ma trận có thể không chéo hóa được trên R")


def _group_eigenvalues(values: list[float], tol: float = 1e-7) -> list[tuple[float, int]]:
    """
    Nhóm các giá trị riêng gần bằng nhau.

    Tham số:
        values: Danh sách các giá trị riêng
        tol: Độ dung sai

    Trả về:
        Danh sách các tuple (giá trị riêng, số lần xuất hiện)
    """
    if not values:
        return []

    sorted_values = sorted(values, reverse=True)
    groups: list[list[float]] = [[sorted_values[0]]]

    for value in sorted_values[1:]:
        if abs(value - groups[-1][-1]) <= tol:
            groups[-1].append(value)
        else:
            groups.append([value])

    result: list[tuple[float, int]] = []
    for g in groups:
        center = sum(g) / len(g)
        result.append((center, len(g)))
    return result


def _build_p_from_columns(cols: list[list[float]]) -> list[list[float]]:
    n = len(cols)
    return [[cols[col][row] for col in range(n)] for row in range(n)]


def diagonalize(A: list[list[float]]) -> tuple[list[list[float]], list[float]]:
    """
    Chéo hóa ma trận A thành A = P D P^{-1}.

    Trả về:
        (P, D)
        - P là ma trận vector riêng, mỗi cột là 1 vector riêng
        - D là danh sách trị riêng 1 chiều, tương ứng theo thứ tự cột của P
    """
    _validate_square_matrix(A)

    A_num = [[float(value) for value in row] for row in A]
    n = len(A_num)

    diag_estimates = _qr_eigen_diagonal(A_num)
    eig_groups = _group_eigenvalues(diag_estimates)

    eigenvectors: list[list[float]] = []
    eigenvalues: list[float] = []

    for lam, multiplicity in eig_groups:
        B = [row[:] for row in A_num]
        for i in range(n):
            B[i][i] = B[i][i] - lam

        _, _, _, raw_basis = rank_and_basis(B, tol=1e-7)
        basis = [normalize(v) for v in raw_basis]
        if len(basis) < multiplicity:
            raise ValueError("Ma trận không chéo hóa được (thiếu vector riêng độc lập)")

        added = 0
        for vec in basis:
            candidate_cols = eigenvectors + [vec]
            P_candidate = [[candidate_cols[col][row] for col in range(len(candidate_cols))] for row in range(n)]
            if rank_and_basis(P_candidate, tol=1e-7)[0] > len(eigenvectors):
                eigenvectors.append(vec)
                eigenvalues.append(lam)
                added += 1
                if added == multiplicity:
                    break

        if added < multiplicity:
            raise ValueError("Ma trận không chéo hóa được (không chọn đủ vector riêng)")

    if len(eigenvectors) != n:
        raise ValueError("Ma trận không chéo hóa được (thiếu cơ sở vector riêng)")

    P = _build_p_from_columns(eigenvectors)
    if rank_and_basis(P, tol=1e-7)[0] != n:
        raise ValueError("Ma trận không chéo hóa được (P không khả nghịch)")

    AP = matmul(A_num, P)
    PD = [[P[i][j] * eigenvalues[j] for j in range(n)] for i in range(n)]
    residual = max_abs_diff(AP, PD)
    if residual > 1e-6:
        raise ValueError("Ma trận không chéo hóa được trên R hoặc sai số số học quá lớn")

    P = rectify_matrix(P)
    D = rectify_vector(eigenvalues)
    return P, D


def _max_relation_error(A: list[list[float]], P: list[list[float]], D: list[float]) -> float:
    AP = matmul(A, P)
    PD = [[P[i][j] * D[j] for j in range(len(D))] for i in range(len(P))]
    return max_abs_diff(AP, PD)


def test_diagonalize(test_cases: list[dict]) -> None:
    """
    Bộ kiểm thử cho diagonalize với nhiều dạng ma trận có thể gặp.
    """
    TestLogger.print_suite_header("Chéo Hóa Ma Trận")
    
    passed_count = 0
    total_count = len(test_cases)

    for case in test_cases:
        try:
            A = case["input"]
            P, D = diagonalize(A)

            if case.get("should_raise") == ValueError:
                TestLogger.print_result(case['name'], False, "Lẽ ra phải phát sinh ValueError")
                continue

            n = len(A)
            assert len(P) == n and len(P[0]) == n, "P phải có kích thước n x n"
            assert len(D) == n, "D phải có đúng n trị riêng"
            assert rank_and_basis(P, tol=1e-7)[0] == n, "P phải khả nghịch (hạng đầy đủ)"

            A_float = [[float(value) for value in row] for row in A]
            err = _max_relation_error(A_float, P, D)
            tol = case.get("tol", 1e-6)
            assert err <= tol, f"Sai số A*P - P*D quá lớn: {err} > {tol}"

            TestLogger.print_result(case['name'], True, f"(err = {err:.2e})")
            passed_count += 1
        except ValueError as e:
            if case.get("should_raise") == ValueError:
                TestLogger.print_result(case['name'], True, f"(Bắt đúng lỗi mong đợi: {e})")
                passed_count += 1
            else:
                TestLogger.print_result(case['name'], False, f"(Lỗi ngoài mong đợi: {e})")
        except AssertionError as e:
            TestLogger.print_result(case['name'], False, f"(Assertion: {e})")
    
    TestLogger.print_summary(passed_count, total_count)

if __name__ == "__main__":
    test_diagonalize(DIAGONALIZATION_TEST_CASES)
