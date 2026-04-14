"""
solvers.py - Giải hệ phương trình tuyến tính Ax = b.

Chiến lược (100% NumPy cho Gauss & SVD, Pure Python cho Gauss-Seidel):
  - 'gauss'        : np.linalg.solve → fallback np.linalg.lstsq nếu singular.
  - 'svd'          : np.linalg.lstsq (pseudo-inverse).
  - 'gauss_seidel' : Vòng lặp Pure Python (max_iter=1000). Yêu cầu chéo trội.
"""
from __future__ import annotations

import math
import numpy as np


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def check_diagonally_dominant(A: list[list[float]]) -> bool:
    """Kiểm tra ma trận có chéo trội nghiêm ngặt theo hàng không."""
    n = len(A)
    for i in range(n):
        diag = abs(A[i][i])
        off_sum = sum(abs(A[i][j]) for j in range(n) if j != i)
        if diag <= off_sum:
            return False
    return True


def _relative_error(A: list[list[float]], x: list[float], b: list[float]) -> float:
    """Sai số tương đối ||Ax - b||₂ / ||b||₂."""
    n = len(b)
    residual = [
        sum(A[i][j] * x[j] for j in range(n)) - b[i]
        for i in range(n)
    ]
    norm_r = math.sqrt(sum(r ** 2 for r in residual))
    norm_b = math.sqrt(sum(bi ** 2 for bi in b))
    return norm_r / norm_b if norm_b != 0.0 else float("inf")


# ---------------------------------------------------------------------------
# Core dispatcher
# ---------------------------------------------------------------------------

def solve_system(
    A: list[list[float]], b: list[float], method: str
) -> list[float]:
    """
    Giải hệ Ax = b theo phương pháp chỉ định.

    Args:
        A      : Ma trận hệ số n × n (list of lists).
        b      : Vector vế phải n phần tử.
        method : 'gauss' | 'svd' | 'gauss_seidel'

    Returns:
        x : Vector nghiệm n phần tử.

    Raises:
        ValueError : Gauss-Seidel được gọi với ma trận không chéo trội.
        ValueError : method không hợp lệ.
    """
    n = len(A)
    A_np = np.array(A, dtype=float)
    b_np = np.array(b, dtype=float)

    # ------------------------------------------------------------------
    # GAUSS (NumPy LU solver + lstsq fallback)
    # ------------------------------------------------------------------
    if method == "gauss":
        try:
            x_np = np.linalg.solve(A_np, b_np)
        except np.linalg.LinAlgError:
            # Ma trận singular (thường gặp với Hilbert lớn) → lstsq fallback
            # Nghiệm least-squares vẫn cho sai số bùng nổ đưa vào báo cáo.
            x_np, _, _, _ = np.linalg.lstsq(A_np, b_np, rcond=None)
        return x_np.tolist()

    # ------------------------------------------------------------------
    # SVD (NumPy pseudo-inverse)
    # ------------------------------------------------------------------
    elif method == "svd":
        x_np, _, _, _ = np.linalg.lstsq(A_np, b_np, rcond=None)
        return x_np.tolist()

    # ------------------------------------------------------------------
    # GAUSS-SEIDEL (Pure Python iterative, max_iter=1000)
    # ------------------------------------------------------------------
    elif method == "gauss_seidel":
        if not check_diagonally_dominant(A):
            raise ValueError(
                "Ma trận không chéo trội nghiêm ngặt → Gauss-Seidel không đảm bảo hội tụ."
            )

        max_iter = 1000
        tol = 1e-10
        x = [0.0] * n

        for _ in range(max_iter):
            x_old = x[:]
            for i in range(n):
                sigma = sum(A[i][j] * x[j] for j in range(n) if j != i)
                x[i] = (b[i] - sigma) / A[i][i]

            diff = math.sqrt(sum((x[k] - x_old[k]) ** 2 for k in range(n)))
            if diff < tol:
                break

        return x

    # ------------------------------------------------------------------
    else:
        raise ValueError(
            f"Phương pháp '{method}' không được hỗ trợ. "
            "Chọn: 'gauss', 'svd', 'gauss_seidel'."
        )
