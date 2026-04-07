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

from config import EPSILON, is_zero, zero_rectify


def _validate_matrix(A: list[list[float]]) -> None:
    if not A:
        raise ValueError("Ma trận A không được rỗng")

    n_cols = len(A[0])
    if n_cols == 0:
        raise ValueError("Ma trận A phải có ít nhất 1 cột")

    for row in A:
        if len(row) != n_cols:
            raise ValueError("Tất cả các dòng của A phải cùng số cột")


def _identity(n: int) -> list[list[float]]:
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def _transpose(A: list[list[float]]) -> list[list[float]]:
    return [list(col) for col in zip(*A)]


def _dot(u: list[float], v: list[float]) -> float:
    return sum(ui * vi for ui, vi in zip(u, v))


def _norm(v: list[float]) -> float:
    return math.sqrt(max(0.0, _dot(v, v)))


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


def _matvec(A: list[list[float]], v: list[float]) -> list[float]:
    if len(A[0]) != len(v):
        raise ValueError("Không thể nhân ma trận-véc tơ: kích thước không tương thích")
    return [sum(aij * vj for aij, vj in zip(row, v)) for row in A]


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


def _jacobi_eigen_symmetric(S: list[list[float]]) -> tuple[list[float], list[list[float]]]:
    n = len(S)
    D = [row[:] for row in S]
    V = _identity(n)

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
        col = _orthogonalize(col, cols)
        col_norm = _norm(col)
        if is_zero(col_norm):
            col = _find_new_unit_vector(cols, n)
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

    A_T = _transpose(A_num)
    A_T_A = _matmul(A_T, A_num)
    eigenvalues, V = _jacobi_eigen_symmetric(A_T_A)

    sigma_all: list[float] = []
    for lam in eigenvalues:
        if lam < 0.0:
            lam = 0.0 if is_zero(lam) else 0.0
        sigma_all.append(math.sqrt(lam) if lam > 0.0 else 0.0)

    Sigma = _rectify_vector(sigma_all[:r])

    u_cols: list[list[float]] = []
    for i in range(r):
        sigma_i = Sigma[i]
        if not is_zero(sigma_i):
            v_i = [V[row][i] for row in range(n)]
            candidate = _matvec(A_num, v_i)
            candidate = [value / sigma_i for value in candidate]
            candidate = _orthogonalize(candidate, u_cols)
            norm_candidate = _norm(candidate)
            if not is_zero(norm_candidate):
                u_cols.append([zero_rectify(x / norm_candidate) for x in candidate])
                continue

        u_cols.append(_find_new_unit_vector(u_cols, m))

    while len(u_cols) < m:
        u_cols.append(_find_new_unit_vector(u_cols, m))

    U = [[u_cols[col][row] for col in range(m)] for row in range(m)]
    V_T = _transpose(V)

    U = _rectify_matrix(U)
    V_T = _rectify_matrix(V_T)
    return U, Sigma, V_T


def _sigma_to_matrix(sigma: list[float], m: int, n: int) -> list[list[float]]:
    S = [[0.0 for _ in range(n)] for _ in range(m)]
    r = min(len(sigma), m, n)
    for i in range(r):
        S[i][i] = sigma[i]
    return S


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


def _is_descending_nonnegative(values: list[float], tol: float = 1e-9) -> bool:
    for v in values:
        if v < -tol:
            return False

    for i in range(len(values) - 1):
        if values[i] + tol < values[i + 1]:
            return False
    return True


def _is_orthogonal(Q: list[list[float]], tol: float = 1e-7) -> bool:
    if not Q:
        return True

    n = len(Q)
    if any(len(row) != n for row in Q):
        return False

    QT = _transpose(Q)
    QTQ = _matmul(QT, Q)
    I = _identity(n)
    err = _max_abs_diff(QTQ, I)
    return err <= tol


def test_decompose_svd() -> None:
    """
    Bộ kiểm thử cho decompose_svd theo nhiều dạng ma trận thường gặp.
    """
    test_cases = [
        {
            "name": "Ma trận vuông full-rank 3x3",
            "input": [[4.0, 1.0, 2.0], [0.0, 3.0, -1.0], [2.0, 0.0, 1.0]],
            "tol": 1e-7,
        },
        {
            "name": "Ma trận chữ nhật ngang 2x3",
            "input": [[3.0, 1.0, 1.0], [-1.0, 3.0, 1.0]],
            "tol": 1e-7,
        },
        {
            "name": "Ma trận chữ nhật dọc 3x2",
            "input": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            "tol": 1e-7,
        },
        {
            "name": "Ma trận suy biến (hai dòng phụ thuộc)",
            "input": [[1.0, 2.0], [2.0, 4.0]],
            "tol": 1e-7,
        },
        {
            "name": "Ma trận toàn 0 kích thước 3x4",
            "input": [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
            "tol": 1e-8,
        },
        {
            "name": "Ma trận đơn vị 4x4",
            "input": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            "tol": 1e-8,
        },
        {
            "name": "Ma trận đường chéo có singular values lặp",
            "input": [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 1.0]],
            "tol": 1e-8,
        },
        {
            "name": "Ma trận chứa số âm",
            "input": [[-3.0, 2.0], [4.0, -1.0]],
            "tol": 1e-7,
        },
        {
            "name": "Ma trận kích thước 1x1",
            "input": [[-7.0]],
            "tol": 1e-8,
        },
        {
            "name": "Ma trận 1xN",
            "input": [[2.0, -1.0, 3.0, 0.5]],
            "tol": 1e-7,
        },
        {
            "name": "Ma trận Nx1",
            "input": [[2.0], [-1.0], [3.0], [0.5]],
            "tol": 1e-7,
        },
        {
            "name": "Ma trận số rất nhỏ gần EPSILON",
            "input": [[1e-13, 0.0], [0.0, 2e-13]],
            "tol": 1e-10,
        },
        {
            "name": "Dữ liệu rỗng",
            "input": [],
            "should_raise": ValueError,
        },
        {
            "name": "Ma trận có 0 cột",
            "input": [[]],
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
            U, sigma, V_T = decompose_svd(A)

            if case.get("should_raise") == ValueError:
                print("=> FAILED: Lẽ ra phải phát sinh ValueError")
                continue

            m = len(A)
            n = len(A[0])
            r = min(m, n)

            assert len(U) == m and len(U[0]) == m, "U phải có kích thước m x m"
            assert len(sigma) == r, "Sigma phải có độ dài min(m, n)"
            assert len(V_T) == n and len(V_T[0]) == n, "V_T phải có kích thước n x n"

            assert _is_descending_nonnegative(sigma), "Sigma phải không âm và giảm dần"
            assert _is_orthogonal(U), "U phải là ma trận trực giao"

            V = _transpose(V_T)
            assert _is_orthogonal(V), "V phải là ma trận trực giao"

            S = _sigma_to_matrix(sigma, m, n)
            A_reconstructed = _matmul(_matmul(U, S), V_T)
            A_float = [[float(value) for value in row] for row in A]

            err = _max_abs_diff(A_reconstructed, A_float)
            tol = case.get("tol", 1e-7)
            assert err <= tol, f"Sai số tái tạo quá lớn: {err} > {tol}"

            print(f"=> PASSED (sai số lớn nhất = {err:.3e})")
        except ValueError as e:
            if case.get("should_raise") == ValueError:
                print(f"=> PASSED (Bắt đúng lỗi mong đợi: {e})")
            else:
                print(f"=> FAILED: Lỗi ngoài mong đợi: {e}")