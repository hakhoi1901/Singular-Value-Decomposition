"""
Cài đặt các phương pháp giải hệ phương trình tuyến tính Ax = b.

Bao gồm:
    - Thuật toán lặp Gauss-Seidel (kèm kiểm tra điều kiện hội tụ).
    - Wrapper giải bằng khử Gauss (từ part1).
    - Wrapper giải bằng ma trận nghịch đảo (từ part1).

Không sử dụng numpy — toàn bộ tính toán bằng Python thuần (list of lists).
"""
from __future__ import annotations

import sys
import io
from contextlib import redirect_stdout
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_PART1 = _ROOT / "part1"
if str(_PART1) not in sys.path:
    sys.path.insert(0, str(_PART1))

from part1.gaussian import gaussian_eliminate
from part1.inverse import inverse
from utils import matvec, vector_norm


def is_strictly_diagonally_dominant(A: list[list[float]]) -> bool:
    """
    Kiểm tra ma trận A có phải là ma trận chéo trội chặt hàng hay không.

    Điều kiện: Với mọi dòng i, |A[i][i]| > Σ_{j≠i} |A[i][j]|.
    Đây là điều kiện đủ để thuật toán Gauss-Seidel hội tụ.

    Tham số:
        A: Ma trận vuông kích thước n × n.

    Trả về:
        bool: True nếu A chéo trội chặt hàng, False ngược lại.
    """
    n = len(A)
    for i in range(n):
        diag_val = abs(A[i][i])
        row_sum = sum(abs(A[i][j]) for j in range(n) if j != i)
        if diag_val <= row_sum:
            return False
    return True


def solve_system_gauss_seidel(
    A: list[list[float]],
    b: list[float],
    max_iters: int = 1000,
    tol: float = 1e-8,
) -> list[float]:
    """
    Giải hệ phương trình tuyến tính Ax = b bằng thuật toán lặp Gauss-Seidel.

    Thuật toán:
        1. Kiểm tra điều kiện hội tụ (ma trận chéo trội chặt hàng).
        2. Khởi tạo nghiệm x⁰ = vector 0.
        3. Tại mỗi bước lặp k, cập nhật từng thành phần:
            x_i^(k+1) = (b_i - Σ_{j<i} A_ij * x_j^(k+1) - Σ_{j>i} A_ij * x_j^(k)) / A_ii
        4. Dừng sớm khi ||x^(k+1) - x^(k)||₂ < tol.

    Tham số:
        A: Ma trận hệ số vuông kích thước n × n.
        b: Vector vế phải kích thước n.
        max_iters: Số bước lặp tối đa (mặc định 1000).
        tol: Ngưỡng hội tụ (mặc định 1e-8).

    Trả về:
        list[float]: Vector nghiệm xấp xỉ x kích thước n.

    Xử lý ngoại lệ:
        - In cảnh báo nếu ma trận không chéo trội chặt (có thể không hội tụ).
    """
    n = len(A)

    # Kiểm tra điều kiện hội tụ
    if not is_strictly_diagonally_dominant(A):
        print("Cảnh báo: Ma trận không chéo trội chặt — Gauss-Seidel có thể không hội tụ.")

    # Khởi tạo nghiệm ban đầu là vector 0
    x = [0.0] * n

    for _ in range(max_iters):
        x_new = list(x)
        for i in range(n):
            # Tính tổng phần đã cập nhật (j < i) và chưa cập nhật (j > i)
            s1 = sum(A[i][j] * x_new[j] for j in range(i))
            s2 = sum(A[i][j] * x[j] for j in range(i + 1, n))

            x_new[i] = (b[i] - s1 - s2) / A[i][i]

        # Kiểm tra hội tụ: ||x_new - x||₂ < tol
        diff = [x_new[i] - x[i] for i in range(n)]
        if vector_norm(diff) < tol:
            return x_new
        x = x_new

    return x


def solve_system_gaussian(A: list[list[float]], b: list[float]) -> list[float]:
    """
    Giải hệ Ax = b bằng phương pháp khử Gauss với partial pivoting.

    Thuật toán:
        Gọi hàm gaussian_eliminate từ part1 để đưa [A|b] về dạng bậc thang
        rồi thế ngược tìm nghiệm.

    Tham số:
        A: Ma trận hệ số vuông kích thước n × n.
        b: Vector vế phải kích thước n.

    Trả về:
        list[float]: Vector nghiệm x kích thước n.
    """
    # Bọc stdout để tránh in thông báo pivot rác ra console
    trap = io.StringIO()
    with redirect_stdout(trap):
        _, x, _ = gaussian_eliminate(A, b)
    return x


def solve_system_inverse(A: list[list[float]], b: list[float]) -> list[float]:
    """
    Giải hệ Ax = b bằng cách tính ma trận nghịch đảo: x = A⁻¹ · b.

    Thuật toán:
        1. Tính A⁻¹ bằng phương pháp Gauss-Jordan (từ part1).
        2. Nhân x = A⁻¹ · b bằng phép nhân ma trận-vector.

    Tham số:
        A: Ma trận hệ số vuông khả nghịch kích thước n × n.
        b: Vector vế phải kích thước n.

    Trả về:
        list[float]: Vector nghiệm x kích thước n.
    """
    A_inv = inverse(A)
    return matvec(A_inv, b)
