import random
import math

def generate_random_system(n: int, force_diagonally_dominant: bool = False):
    """
    Sinh ma trận A ngẫu nhiên và vector b.
    Nếu force_diagonally_dominant=True, ép chéo trội để Gauss-Seidel hội tụ.
    Trả về: (A, b)
    """
    A = [[random.uniform(0, 10) for _ in range(n)] for _ in range(n)]
    if force_diagonally_dominant:
        for i in range(n):
            row_sum = sum(abs(A[i][j]) for j in range(n)) - abs(A[i][i])
            A[i][i] = row_sum + random.uniform(1, 5)
    b = [random.uniform(0, 10) for _ in range(n)]
    return A, b


def generate_spd_matrix(n: int) -> list[list[float]]:
    """
    Sinh ma trận đối xứng xác định dương (Well-conditioned).
    Thuật toán: A = X * X^T + n * I  (X ngẫu nhiên).
    """
    import numpy as np
    X = np.random.rand(n, n)
    A_np = X @ X.T + n * np.eye(n)
    return A_np.tolist()


def generate_hilbert_matrix(n: int) -> list[list[float]]:
    """
    Sinh ma trận Hilbert H_n (Ill-conditioned).
    Công thức: H[i][j] = 1 / (i + j + 1)
    """
    return [[1.0 / (i + j + 1) for j in range(n)] for i in range(n)]


def calculate_condition_number(A: list[list[float]]) -> float:
    """
    Tính số điều kiện κ₂(A) = ||A||₂ · ||A⁻¹||₂  (chuẩn spectral).
    """
    import numpy as np
    return float(np.linalg.cond(A, p=2))