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

from config import EPSILON, is_zero, zero_rectify


def _validate_square_matrix(A: list[list[float]]) -> None:
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


def _identity(n: int) -> list[list[float]]:
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def _transpose(A: list[list[float]]) -> list[list[float]]:
    return [list(col) for col in zip(*A)]


def _dot(u: list[float], v: list[float]) -> float:
    return sum(ui * vi for ui, vi in zip(u, v))


def _norm(v: list[float]) -> float:
    return math.sqrt(max(0.0, _dot(v, v)))


def _is_zero_tol(x: float, tol: float) -> bool:
    return is_zero(x) or abs(x) <= tol


def _rectify_matrix(M: list[list[float]]) -> list[list[float]]:
    rectified: list[list[float]] = []
    for row in M:
        rectified_row: list[float] = []
        for value in row:
            v = zero_rectify(float(value))
            if is_zero(v):
                v = 0.0
            rectified_row.append(v)
        rectified.append(rectified_row)
    return rectified


def _rectify_vector(v: list[float]) -> list[float]:
    rectified: list[float] = []
    for value in v:
        x = zero_rectify(float(value))
        if is_zero(x):
            x = 0.0
        rectified.append(x)
    return rectified


def _matmul(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    m = len(A)
    p = len(A[0])
    if len(B) != p:
        raise ValueError("Không thể nhân ma trận: kích thước không tương thích")

    n = len(B[0])
    C = [[0.0 for _ in range(n)] for _ in range(m)]
    for i in range(m):
        for k in range(p):
            aik = A[i][k]
            if is_zero(aik):
                continue
            for j in range(n):
                C[i][j] += aik * B[k][j]
    return C


def _max_abs_diff(A: list[list[float]], B: list[list[float]]) -> float:
    if len(A) != len(B):
        raise ValueError("Hai ma trận phải có cùng số dòng")
    if A and len(A[0]) != len(B[0]):
        raise ValueError("Hai ma trận phải có cùng số cột")

    max_err = 0.0
    for i in range(len(A)):
        for j in range(len(A[0])):
            diff = abs(A[i][j] - B[i][j])
            if diff > max_err:
                max_err = diff
    return max_err


def _orthogonalize(v: list[float], basis: list[list[float]]) -> list[float]:
    w = [float(x) for x in v]
    for b in basis:
        coeff = _dot(w, b)
        if is_zero(coeff):
            continue
        for i in range(len(w)):
            w[i] = zero_rectify(w[i] - coeff * b[i])
    return w


def _find_new_unit_vector(basis: list[list[float]], dim: int) -> list[float]:
    for idx in range(dim):
        candidate = [0.0] * dim
        candidate[idx] = 1.0
        w = _orthogonalize(candidate, basis)
        nw = _norm(w)
        if not is_zero(nw):
            return [zero_rectify(x / nw) for x in w]

    candidate = [float(i + 1) for i in range(dim)]
    w = _orthogonalize(candidate, basis)
    nw = _norm(w)
    if is_zero(nw):
        raise ValueError("Không thể xây dựng cơ sở trực giao")
    return [zero_rectify(x / nw) for x in w]


def _normalize(v: list[float]) -> list[float]:
    nv = _norm(v)
    if is_zero(nv):
        return [0.0 for _ in v]
    return [zero_rectify(x / nv) for x in v]


def _qr_decomposition(A: list[list[float]]) -> tuple[list[list[float]], list[list[float]]]:
    n = len(A)
    q_cols: list[list[float]] = []
    R = [[0.0 for _ in range(n)] for _ in range(n)]

    for j in range(n):
        v = [A[i][j] for i in range(n)]

        for i in range(j):
            q_i = q_cols[i]
            r_ij = _dot(q_i, v)
            R[i][j] = r_ij
            if not is_zero(r_ij):
                for k in range(n):
                    v[k] = zero_rectify(v[k] - r_ij * q_i[k])

        r_jj = _norm(v)
        if is_zero(r_jj):
            q_j = _find_new_unit_vector(q_cols, n)
            R[j][j] = 0.0
        else:
            q_j = [zero_rectify(x / r_jj) for x in v]
            R[j][j] = r_jj

        q_cols.append(q_j)

    Q = [[q_cols[col][row] for col in range(n)] for row in range(n)]
    Q = _rectify_matrix(Q)
    R = _rectify_matrix(R)
    return Q, R


def _qr_eigen_diagonal(A: list[list[float]], tol: float = 1e-10, max_iter: int = 4000) -> list[float]:
    n = len(A)
    A_k = [row[:] for row in A]

    for _ in range(max_iter):
        Q, R = _qr_decomposition(A_k)
        A_k = _matmul(R, Q)
        A_k = _rectify_matrix(A_k)

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


def _rref(A: list[list[float]], tol: float = 1e-8) -> tuple[list[list[float]], list[int]]:
    if not A:
        return [], []

    n_rows = len(A)
    n_cols = len(A[0])
    M = [row[:] for row in A]

    pivot_row = 0
    pivot_cols: list[int] = []

    for j in range(n_cols):
        if pivot_row >= n_rows:
            break

        selected = pivot_row
        best = abs(M[pivot_row][j])
        for r in range(pivot_row + 1, n_rows):
            value = abs(M[r][j])
            if value > best:
                best = value
                selected = r

        if _is_zero_tol(M[selected][j], tol):
            continue

        M[pivot_row], M[selected] = M[selected], M[pivot_row]
        pivot = M[pivot_row][j]

        for c in range(j, n_cols):
            M[pivot_row][c] = M[pivot_row][c] / pivot
            if _is_zero_tol(M[pivot_row][c], tol):
                M[pivot_row][c] = 0.0

        for r in range(n_rows):
            if r == pivot_row:
                continue
            factor = M[r][j]
            if _is_zero_tol(factor, tol):
                continue

            for c in range(j, n_cols):
                M[r][c] = M[r][c] - factor * M[pivot_row][c]
                if _is_zero_tol(M[r][c], tol):
                    M[r][c] = 0.0

        pivot_cols.append(j)
        pivot_row += 1

    return M, pivot_cols


def _null_space_basis(A: list[list[float]], tol: float = 1e-8) -> list[list[float]]:
    if not A:
        return []

    n_cols = len(A[0])
    rref, pivot_cols = _rref(A, tol)
    free_cols = [j for j in range(n_cols) if j not in pivot_cols]

    if not free_cols:
        return []

    basis: list[list[float]] = []
    for free_j in free_cols:
        vec = [0.0] * n_cols
        vec[free_j] = 1.0

        for i, p_col in enumerate(pivot_cols):
            vec[p_col] = zero_rectify(-rref[i][free_j])

        basis.append(_normalize(vec))

    return basis


def _rank(A: list[list[float]], tol: float = 1e-8) -> int:
    if not A:
        return 0
    _, pivot_cols = _rref(A, tol)
    return len(pivot_cols)


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

        basis = _null_space_basis(B)
        if len(basis) < multiplicity:
            raise ValueError("Ma trận không chéo hóa được (thiếu vector riêng độc lập)")

        added = 0
        for vec in basis:
            candidate_cols = eigenvectors + [vec]
            P_candidate = [[candidate_cols[col][row] for col in range(len(candidate_cols))] for row in range(n)]
            if _rank(P_candidate) > len(eigenvectors):
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
    if _rank(P) != n:
        raise ValueError("Ma trận không chéo hóa được (P không khả nghịch)")

    AP = _matmul(A_num, P)
    PD = [[P[i][j] * eigenvalues[j] for j in range(n)] for i in range(n)]
    residual = _max_abs_diff(AP, PD)
    if residual > 1e-6:
        raise ValueError("Ma trận không chéo hóa được trên R hoặc sai số số học quá lớn")

    P = _rectify_matrix(P)
    D = _rectify_vector(eigenvalues)
    return P, D


def _max_relation_error(A: list[list[float]], P: list[list[float]], D: list[float]) -> float:
    AP = _matmul(A, P)
    PD = [[P[i][j] * D[j] for j in range(len(D))] for i in range(len(P))]
    return _max_abs_diff(AP, PD)


def test_diagonalize() -> None:
    """
    Bộ kiểm thử cho diagonalize với nhiều dạng ma trận có thể gặp.
    """
    test_cases = [
        {
            "name": "Ma trận chéo 3x3, trị riêng phân biệt",
            "input": [[5.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, -1.0]],
            "tol": 1e-8,
        },
        {
            "name": "Ma trận tam giác trên có trị riêng thực phân biệt",
            "input": [[4.0, 1.0, 0.0], [0.0, 2.0, 1.0], [0.0, 0.0, -3.0]],
            "tol": 1e-6,
        },
        {
            "name": "Ma trận đối xứng có trị riêng lặp",
            "input": [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]],
            "tol": 1e-8,
        },
        {
            "name": "Ma trận đơn vị 4x4",
            "input": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            "tol": 1e-8,
        },
        {
            "name": "Ma trận suy biến nhưng chéo hóa được",
            "input": [[0.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]],
            "tol": 1e-8,
        },
        {
            "name": "Ma trận kích thước 1x1",
            "input": [[-7.0]],
            "tol": 1e-10,
        },
        {
            "name": "Khối Jordan bậc 2 (không chéo hóa được)",
            "input": [[2.0, 1.0], [0.0, 2.0]],
            "should_raise": ValueError,
        },
        {
            "name": "Ma trận quay có trị riêng phức",
            "input": [[0.0, -1.0], [1.0, 0.0]],
            "should_raise": ValueError,
        },
        {
            "name": "Ma trận rỗng",
            "input": [],
            "should_raise": ValueError,
        },
        {
            "name": "Ma trận không vuông",
            "input": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            "should_raise": ValueError,
        },
        {
            "name": "Ma trận không cùng số cột",
            "input": [[1.0, 2.0], [3.0]],
            "should_raise": ValueError,
        },
    ]

    for case in test_cases:
        print(f"Kiểm thử: {case['name']}")
        try:
            A = case["input"]
            P, D = diagonalize(A)

            if case.get("should_raise") == ValueError:
                print("=> FAILED: Lẽ ra phải phát sinh ValueError")
                continue

            n = len(A)
            assert len(P) == n and len(P[0]) == n, "P phải có kích thước n x n"
            assert len(D) == n, "D phải có đúng n trị riêng"
            assert _rank(P) == n, "P phải khả nghịch (hạng đầy đủ)"

            A_float = [[float(value) for value in row] for row in A]
            err = _max_relation_error(A_float, P, D)
            tol = case.get("tol", 1e-6)
            assert err <= tol, f"Sai số A*P - P*D quá lớn: {err} > {tol}"

            print(f"=> PASSED (sai số lớn nhất = {err:.3e})")
        except ValueError as e:
            if case.get("should_raise") == ValueError:
                print(f"=> PASSED (Bắt đúng lỗi mong đợi: {e})")
            else:
                print(f"=> FAILED: Lỗi ngoài mong đợi: {e}")

if __name__ == "__main__":
    test_diagonalize()
