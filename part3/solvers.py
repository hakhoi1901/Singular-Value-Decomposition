"""
solvers.py - File thuật toán giải hệ phương trình tuyến tính.
Cung cấp hàm phân luồng solve_system() theo spec §II.2.
"""
from __future__ import annotations

import sys
import math
from pathlib import Path

# ----- Path bootstrap -----
_PART3 = Path(__file__).resolve().parent          # .../part3
_ROOT  = _PART3.parent                             # .../VSCodeTUDTK
_PART1 = _ROOT / "part1"
_PART2 = _ROOT / "part2"

for _p in [str(_ROOT), str(_PART1), str(_PART2)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import EPSILON, is_zero, calculate_relative_error


def check_diagonally_dominant(A: list[list[float]]) -> bool:
    """Kiểm tra ma trận có chéo trội chặt hàng không."""
    n = len(A)
    for i in range(n):
        diag = abs(A[i][i])
        row_sum = sum(abs(A[i][j]) for j in range(n) if j != i)
        if diag <= row_sum:
            return False
    return True


def solve_system(A: list[list[float]], b: list[float], method: str) -> list[float]:
    """
    Hàm phân luồng.
        A: Ma trận hệ số kích thước n x n.
        b: Vector vế phải kích thước n.
        method: Tên phương pháp giải, thuộc ['gauss', 'svd', 'gauss_seidel'].
    Returns:
        x: Vector nghiệm kích thước n.
    Raises:
        ValueError: Nếu ma trận suy biến (cho Gauss) hoặc không thỏa điều kiện hội tụ (cho Gauss-Seidel).
    """
    n = len(A)

    if method == 'gauss':
        from gaussian import gaussian_eliminate
        _, x, _ = gaussian_eliminate(A, b)
        if not x:
            raise ValueError("Ma trận suy biến hoặc hệ không có nghiệm duy nhất (Gauss).")
        return x

    elif method == 'svd':
        from decomposition import decompose_svd
        U, sigma, V_T = decompose_svd(A)
        # V = V_T^T
        V = [[V_T[j][i] for j in range(n)] for i in range(n)]
        # U^T * b
        Ut_b = [sum(U[i][k] * b[i] for i in range(n)) for k in range(n)]
        # Sigma^+ * (U^T b): nghịch đảo giả, bỏ qua singular values < EPSILON
        sigma_inv_Ut_b = [
            (Ut_b[i] / sigma[i]) if i < len(sigma) and abs(sigma[i]) > EPSILON else 0.0
            for i in range(n)
        ]
        # x = V * sigma_inv_Ut_b
        x = [sum(V[i][j] * sigma_inv_Ut_b[j] for j in range(n)) for i in range(n)]
        return x

    elif method == 'gauss_seidel':
        if not check_diagonally_dominant(A):
            raise ValueError("Ma trận không chéo trội chặt hàng, Gauss-Seidel có thể không hội tụ.")

        max_iter = 10000
        tol = 1e-10
        x = [0.0] * n

        for _ in range(max_iter):
            x_old = x[:]
            for i in range(n):
                sigma_gs = sum(A[i][j] * x[j] for j in range(n) if j != i)
                x[i] = (b[i] - sigma_gs) / A[i][i]

            # Kiểm tra hội tụ
            diff = math.sqrt(sum((x[i] - x_old[i]) ** 2 for i in range(n)))
            if diff < tol:
                break

        return x

    else:
        raise ValueError(f"Phương pháp '{method}' không được hỗ trợ. Chọn: 'gauss', 'svd', 'gauss_seidel'.")
