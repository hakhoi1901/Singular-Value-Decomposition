"""
solvers.py - Bộ giải hệ phương trình tuyến tính Ax = b (Phần 3).

  - Chứa các hàm Wrapper gọi trực tiếp từ Phần 1 (Pure Python).
  - Chứa thuật toán lặp Gauss-Seidel tự cài đặt.
  - Chứa hàm điều phối `solve_system` dùng NumPy backend để Benchmark các ma trận khổng lồ.
"""
from __future__ import annotations

import sys
import io
import math
import numpy as np
from contextlib import redirect_stdout
from pathlib import Path

# ----- Thiết lập đường dẫn để import từ part1 và part2 -----
_ROOT = Path(__file__).resolve().parent.parent
for _p in [str(_ROOT), str(_ROOT / "part1"), str(_ROOT / "part2")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from part1.gaussian import gaussian_eliminate
    from part1.inverse import inverse
    from config import EPSILON
except ImportError:
    pass # Bỏ qua nếu chạy độc lập không có file config

# ==============================================================================
# PHẦN A: CÁC HÀM CỐT LÕI (CORE FUNCTIONS - PURE PYTHON)
# ==============================================================================

def is_strictly_diagonally_dominant(A: list[list[float]]) -> bool:
    """Kiểm tra ma trận có chéo trội nghiêm ngặt theo hàng không."""
    n = len(A)
    for i in range(n):
        diag_val = abs(A[i][i])
        row_sum = sum(abs(A[i][j]) for j in range(n) if j != i)
        if diag_val <= row_sum:
            return False
    return True

def solve_system_gauss_seidel(
    A: list[list[float]], b: list[float], max_iters: int = 1000, tol: float = 1e-10
) -> list[float]:
    """Giải hệ Ax = b bằng thuật toán lặp Gauss-Seidel (Pure Python)."""
    n = len(A)
    if not is_strictly_diagonally_dominant(A):
        raise ValueError("Ma trận không chéo trội nghiêm ngặt → Gauss-Seidel không đảm bảo hội tụ.")

    x = [0.0] * n
    for _ in range(max_iters):
        x_old = x[:]
        for i in range(n):
            s1 = sum(A[i][j] * x[j] for j in range(i))       # Đã cập nhật
            s2 = sum(A[i][j] * x_old[j] for j in range(i + 1, n)) # Chưa cập nhật
            x[i] = (b[i] - s1 - s2) / A[i][i]

        # Kiểm tra hội tụ chuẩn L2
        diff = math.sqrt(sum((x[i] - x_old[i]) ** 2 for i in range(n)))
        if diff < tol:
            break
    return x

def solve_system_gaussian_pure(A: list[list[float]], b: list[float]) -> list[float]:
    """Wrapper: Giải hệ Ax = b bằng Khử Gauss tự cài đặt từ Phần 1."""
    trap = io.StringIO()
    with redirect_stdout(trap): # Chặn in rác ra màn hình
        _, x, _ = gaussian_eliminate(A, b)
    if not x:
        raise ValueError("Ma trận suy biến (Gaussian Pure).")
    return x

# ==============================================================================
# PHẦN B: HÀM ĐIỀU PHỐI BENCHMARK (DISPATCHER)
# ==============================================================================

def solve_system(A: list[list[float]], b: list[float], method: str) -> list[float]:
    """
    Hàm phân luồng phục vụ riêng cho Benchmark (Phần 3).
    
    Chiến lược (Apples-to-Apples):
      - 'gauss_seidel' : Gọi hàm Pure Python tự code (solve_system_gauss_seidel).
      - 'gauss' / 'svd': Gọi NumPy để đảm bảo không bị Timeout ở n=1000 và 
                         xử lý an toàn ma trận Hilbert.
    """
    n = len(A)
    
    if method == "gauss_seidel":
        return solve_system_gauss_seidel(A, b)

    # Chuyển đổi dữ liệu sang NumPy để Benchmark các phương pháp trực tiếp
    A_np = np.array(A, dtype=float)
    b_np = np.array(b, dtype=float)

    if method == "gauss":
        try:
            x_np = np.linalg.solve(A_np, b_np)
        except np.linalg.LinAlgError:
            # Fallback nếu ma trận Hilbert bị Singular
            x_np, _, _, _ = np.linalg.lstsq(A_np, b_np, rcond=None)
        return x_np.tolist()

    elif method == "svd":
        x_np, _, _, _ = np.linalg.lstsq(A_np, b_np, rcond=None)
        return x_np.tolist()

    # (Tùy chọn) Thêm method 'gauss_pure' nếu nhóm muốn test code Phần 1 trên n nhỏ
    elif method == "gauss_pure":
        return solve_system_gaussian_pure(A, b)

    else:
        raise ValueError(f"Phương pháp '{method}' không được hỗ trợ.")
