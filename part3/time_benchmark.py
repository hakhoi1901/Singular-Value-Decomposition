"""
time_benchmark.py - Đo thời gian thực thi các solver trên cả Hilbert và SPD.

Chiến lược đồng nhất với benchmark.py:
  - 'gauss' / 'svd'        : NumPy (solve + lstsq fallback).
  - 'gauss_seidel'         : Pure Python iterative (chỉ hội tụ với SPD).
  - Mỗi phép đo = trung bình 5 lần gọi (N_RUNS = 5).

Output: Bảng thời gian tách biệt cho SPD và Hilbert, in ra terminal.
"""
from __future__ import annotations

import math
import sys
import time
import warnings
from pathlib import Path

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_PART3 = Path(__file__).resolve().parent
_ROOT  = _PART3.parent
for _p in [str(_ROOT), str(_ROOT / "part1"), str(_ROOT / "part2")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data_gen import generate_hilbert_matrix, generate_spd_matrix
from solvers import solve_system

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SIZES   = [50, 100, 200, 500, 1000]
METHODS = ["gauss", "svd", "gauss_seidel"]
N_RUNS  = 5   # số lần lặp để tính thời gian trung bình


# ---------------------------------------------------------------------------
# Core measurement
# ---------------------------------------------------------------------------

def measure_time(
    A: list[list[float]],
    b: list[float],
    method: str,
    runs: int = N_RUNS,
) -> float | str:
    """
    Gọi solve_system `runs` lần, trả về thời gian trung bình (giây).
    Nếu solver ném ngoại lệ → trả về chuỗi mô tả lỗi ngắn gọn.

    Args:
        A      : Ma trận hệ số.
        b      : Vector vế phải.
        method : 'gauss' | 'svd' | 'gauss_seidel'
        runs   : Số lần lặp (mặc định N_RUNS = 5).

    Returns:
        float  : Thời gian trung bình (s).
        str    : Chuỗi lỗi nếu solver thất bại.
    """
    total = 0.0
    for _ in range(runs):
        t0 = time.perf_counter()
        try:
            solve_system(A, b, method)
        except Exception as exc:
            return f"ERR: {exc}"
        total += time.perf_counter() - t0
    return total / runs


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_time_benchmark() -> list[dict]:
    """
    Chạy benchmark thời gian trên cả Hilbert và SPD với các kích thước SIZES.
    Gauss-Seidel sẽ cho ERR với Hilbert (không chéo trội) — đây là kết quả hợp lệ.

    Returns:
        Danh sách dict: {"n", "type", "gauss", "svd", "gauss_seidel"}
    """
    rows: list[dict] = []

    for n in SIZES:
        for label, gen_fn in [("SPD",     generate_spd_matrix),
                               ("Hilbert", generate_hilbert_matrix)]:
            print(f"  [TIME] n={n:>4} ({label}) ...", flush=True)

            A = gen_fn(n)
            b = [sum(row) for row in A]
            row: dict = {"n": n, "type": label}

            for method in METHODS:
                result = measure_time(A, b, method)
                row[method] = result

            rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Table formatter
# ---------------------------------------------------------------------------

def _fmt(v: object) -> str:
    """Định dạng giá trị ô bảng: float → khoa học, lỗi → ERR."""
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return "INF"
        return f"{v:.4f}s"
    if isinstance(v, str):
        # Rút gọn chuỗi lỗi dài thành "ERR"
        return "ERR"
    return "N/A"


def format_time_table(rows: list[dict]) -> str:
    """
    Định dạng bảng thời gian trung bình với căn lề thẳng tắp.
    Cột: [n] | [Loại] | [Gauss (s)] | [SVD (s)] | [Gauss-Seidel (s)]
    """
    col_n    = 6
    col_type = 8
    col_t    = 16

    header = (
        f"{'n':>{col_n}} | "
        f"{'Loại':>{col_type}} | "
        f"{'Gauss (s)':>{col_t}} | "
        f"{'SVD (s)':>{col_t}} | "
        f"{'Gauss-Seidel (s)':>{col_t}}"
    )
    sep = "─" * len(header)

    lines = [
        "",
        sep,
        " BẢNG THỜI GIAN TRUNG BÌNH (avg 5 lần) — Gauss/SVD: NumPy | GS: Pure Python",
        sep,
        header,
        sep,
    ]

    prev_n = None
    for r in rows:
        # Thêm dòng phân cách giữa các nhóm kích thước
        if prev_n is not None and r["n"] != prev_n:
            lines.append(sep)
        prev_n = r["n"]

        lines.append(
            f"{r['n']:>{col_n}} | "
            f"{r['type']:>{col_type}} | "
            f"{_fmt(r.get('gauss')):>{col_t}} | "
            f"{_fmt(r.get('svd')):>{col_t}} | "
            f"{_fmt(r.get('gauss_seidel')):>{col_t}}"
        )

    lines.append(sep)
    lines.append(" ERR (GS/Hilbert): Ma trận Hilbert không chéo trội → Gauss-Seidel không hội tụ.")
    lines.append(" Đơn vị thời gian: giây (s). Mỗi ô = trung bình 5 lần chạy.")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    print("=" * 65)
    print(" TIME BENCHMARK — ĐO THỜI GIAN THỰC THI CÁC PHƯƠNG PHÁP GIẢI")
    print("=" * 65)
    print(f" Sizes   : {SIZES}")
    print(f" Methods : {METHODS}")
    print(f" Runs    : {N_RUNS} lần mỗi phép đo\n")

    results = run_time_benchmark()
    print(format_time_table(results))
