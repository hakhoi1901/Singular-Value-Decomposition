"""
Thực nghiệm so sánh 3 phương pháp giải hệ phương trình tuyến tính Ax = b.

Các phương pháp:
    1. Khử Gauss (partial pivoting)
    2. Ma trận nghịch đảo (Gauss-Jordan)
    3. Lặp Gauss-Seidel

Quy trình:
    - Sinh ma trận ngẫu nhiên chéo trội chặt kích thước n × n.
    - Với mỗi n ∈ {50, 100, 200, 500, 1000}, chạy 5 lần rồi lấy trung bình.
    - Đo thời gian thực thi và sai số tương đối ||Ax̂ - b||₂ / ||b||₂.
    - Vẽ đồ thị log-log kèm đường lý thuyết O(n³) để đối chứng.

Không sử dụng numpy — toàn bộ tính toán bằng Python thuần (list of lists).
"""
from __future__ import annotations

import sys
import os
import random
import time
import io
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                       # Backend không cần GUI
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils import matvec, vector_norm
from part3.solvers import (
    solve_system_gauss_seidel,
    solve_system_gaussian,
    solve_system_inverse,
)


def generate_diagonally_dominant_matrix(n: int):
    """
    Sinh ma trận A vuông (n × n) chéo trội chặt và vector b.

    Thuật toán:
        1. Sinh ngẫu nhiên A[i][j] ∈ [-10, 10].
        2. Đặt |A[i][i]| = Σ_{j≠i} |A[i][j]| + rand(1, 10) để đảm bảo
           chéo trội chặt hàng (điều kiện đủ cho Gauss-Seidel hội tụ).
        3. Dấu của A[i][i] được chọn ngẫu nhiên.
        4. Sinh vector b ngẫu nhiên.

    Tham số:
        n: Kích thước ma trận (n × n).

    Trả về:
        tuple[list[list[float]], list[float]]: Cặp (A, b).
    """
    A = [[random.uniform(-10, 10) for _ in range(n)] for _ in range(n)]
    for i in range(n):
        row_sum = sum(abs(A[i][j]) for j in range(n) if j != i)
        A[i][i] = row_sum + random.uniform(1.0, 10.0)
        if random.random() < 0.5:
            A[i][i] = -A[i][i]

    b = [random.uniform(-10, 10) for _ in range(n)]
    return A, b


def calc_relative_error(A: list[list[float]], x: list[float], b: list[float]) -> float:
    """
    Tính sai số tương đối của nghiệm x̂ cho hệ Ax = b.

    Công thức: e = ||Ax̂ - b||₂ / ||b||₂

    Tham số:
        A: Ma trận hệ số kích thước n × n.
        x: Vector nghiệm xấp xỉ kích thước n.
        b: Vector vế phải kích thước n.

    Trả về:
        float: Sai số tương đối. Trả về NaN nếu x rỗng.
    """
    if x is None or len(x) == 0:
        return float("NaN")

    Ax = matvec(A, x)
    diff = [Ax[i] - b[i] for i in range(len(b))]
    norm_diff = vector_norm(diff)
    norm_b = vector_norm(b)

    if norm_b == 0:
        return norm_diff
    return norm_diff / norm_b


def run_experiment():
    """
    Chạy toàn bộ thực nghiệm: đo thời gian, tính sai số, in bảng, vẽ đồ thị.
    """
    sizes = [50, 100, 200, 500, 1000]
    num_runs = 5

    times_gaussian: list[float] = []
    times_inverse: list[float] = []
    times_gs: list[float] = []

    errors_gaussian: list[float] = []
    errors_inverse: list[float] = []
    errors_gs: list[float] = []

    print("=" * 88)
    print("Bắt đầu thực nghiệm so sánh thuật toán giải Hệ phương trình (trung bình 5 lần chạy):")
    print("=" * 88)
    print(f"{'Kích thước(n)':^15} | {'Gaussian Time (s)':^20} | {'Inverse Time (s)':^20} | {'Gauss-Seidel Time (s)':^22}")
    print("-" * 88)

    for n in sizes:
        total_time_gauss = 0.0
        total_time_inv = 0.0
        total_time_gs = 0.0

        sum_err_gauss = 0.0
        sum_err_inv = 0.0
        sum_err_gs = 0.0

        for _ in range(num_runs):
            A, b = generate_diagonally_dominant_matrix(n)

            # 1. Khử Gauss
            start = time.perf_counter()
            x_gauss = solve_system_gaussian(A, b)
            end = time.perf_counter()
            total_time_gauss += (end - start)
            sum_err_gauss += calc_relative_error(A, x_gauss, b)

            # 2. Ma trận Nghịch đảo
            start = time.perf_counter()
            x_inv = solve_system_inverse(A, b)
            end = time.perf_counter()
            total_time_inv += (end - start)
            sum_err_inv += calc_relative_error(A, x_inv, b)

            # 3. Gauss-Seidel (bọc stdout để tránh cảnh báo lặp lại)
            trap = io.StringIO()
            with redirect_stdout(trap):
                start = time.perf_counter()
                x_gs = solve_system_gauss_seidel(A, b, max_iters=1000, tol=1e-8)
                end = time.perf_counter()
            total_time_gs += (end - start)
            sum_err_gs += calc_relative_error(A, x_gs, b)

        avg_time_gauss = total_time_gauss / num_runs
        avg_time_inv = total_time_inv / num_runs
        avg_time_gs = total_time_gs / num_runs

        avg_err_gauss = sum_err_gauss / num_runs
        avg_err_inv = sum_err_inv / num_runs
        avg_err_gs = sum_err_gs / num_runs

        times_gaussian.append(avg_time_gauss)
        times_inverse.append(avg_time_inv)
        times_gs.append(avg_time_gs)

        errors_gaussian.append(avg_err_gauss)
        errors_inverse.append(avg_err_inv)
        errors_gs.append(avg_err_gs)

        print(f"{n:^15} | {avg_time_gauss:^20.5f} | {avg_time_inv:^20.5f} | {avg_time_gs:^22.5f}")

    print("-" * 88)
    print("\nChi tiết sai số tương đối (trung bình):")
    for idx, n in enumerate(sizes):
        print(f"N = {n}:")
        print(f"  - Gaussian:     {errors_gaussian[idx]:.2e}")
        print(f"  - Inverse:      {errors_inverse[idx]:.2e}")
        print(f"  - Gauss-Seidel: {errors_gs[idx]:.2e}")

    # ── Vẽ đồ thị log-log ──
    plt.figure(figsize=(10, 6))

    plt.loglog(sizes, times_gaussian, marker="o", label="Khử Gauss")
    plt.loglog(sizes, times_inverse, marker="s", label="Ma trận nghịch đảo")
    plt.loglog(sizes, times_gs, marker="^", label="Gauss-Seidel")

    # Đường lý thuyết O(n³) — lấy chuẩn theo điểm Gaussian cuối
    if times_gaussian[-1] > 0:
        c = times_gaussian[-1] / (sizes[-1] ** 3)
        theoretical = [c * (x ** 3) for x in sizes]
        plt.loglog(sizes, theoretical, linestyle="--", color="k", label=r"Lý thuyết $O(n^3)$")

    plt.xlabel("Kích thước ma trận n (log scale)")
    plt.ylabel("Thời gian thực thi trung bình (giây) (log scale)")
    plt.title("Biểu đồ log-log so sánh thời gian thực thi các thuật toán giải HPT")
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    output_path = os.path.join(os.path.dirname(__file__), "benchmark_plot.png")
    plt.savefig(output_path, dpi=150)
    print(f"\nĐã lưu biểu đồ thực nghiệm vào file: {output_path}")


if __name__ == "__main__":
    run_experiment()
