"""
data_gen.py - Sinh dữ liệu ma trận cho benchmark Phần 3.
Hỗ trợ: SPD (Strictly Diagonally Dominant), Hilbert (Ill-conditioned), Random.
"""
from __future__ import annotations

import numpy as np


def generate_spd_matrix(n: int) -> list[list[float]]:
    """
    Sinh ma trận Symmetric Positive Definite (SPD) và ép Chéo Trội Nghiêm Ngặt.
    Công thức: A = X @ X.T, sau đó với mỗi hàng i:
        A[i][i] += sum(|A[i][j]| cho j != i) + 1.0
    Đảm bảo Gauss-Seidel chắc chắn hội tụ.
    """
    X = np.random.rand(n, n)
    A = X @ X.T

    for i in range(n):
        off_diag_sum = np.sum(np.abs(A[i, :])) - abs(A[i, i])
        A[i, i] = off_diag_sum + 1.0  # +1.0 margin → strictly dominant

    return A.tolist()


def generate_hilbert_matrix(n: int) -> list[list[float]]:
    """
    Sinh ma trận Hilbert H_n (Ill-conditioned).
    Công thức: H[i][j] = 1 / (i + j + 1)  (0-indexed)
    """
    return [[1.0 / (i + j + 1) for j in range(n)] for i in range(n)]


def generate_random_system(
    n: int, force_diagonally_dominant: bool = False
) -> tuple[list[list[float]], list[float]]:
    """
    Sinh hệ phương trình Ax = b ngẫu nhiên.
    Nếu force_diagonally_dominant=True, ép chéo trội nghiêm ngặt (dùng cho Gauss-Seidel).
    Trả về: (A, b)
    """
    import random

    A = [[random.uniform(0, 10) for _ in range(n)] for _ in range(n)]
    if force_diagonally_dominant:
        for i in range(n):
            off_diag_sum = sum(abs(A[i][j]) for j in range(n) if j != i)
            A[i][i] = off_diag_sum + random.uniform(1.0, 5.0)  # +margin

    b = [random.uniform(0, 10) for _ in range(n)]
    return A, b


def calculate_condition_number(A: list[list[float]]) -> float:
    """
    Tính số điều kiện κ₂(A) = ||A||₂ · ||A⁻¹||₂  (chuẩn spectral).
    Dùng np.linalg.cond để tránh tràn số với ma trận lớn.
    """
    return float(np.linalg.cond(np.array(A, dtype=float), p=2))