"""
benchmark.py - Đo lường hiệu năng và độ ổn định của các solver.

Xuất:
  - benchmark_results.json : Dữ liệu thô để vẽ đồ thị.
  - Terminal               : 2 bảng số liệu căn lề rõ ràng.

Cấu trúc JSON:
{
  "performance": {
    "gauss":        { "50": {"time_s": ..., "error": ...}, ... },
    "svd":          { ... },
    "gauss_seidel": { ... }
  },
  "stability": {
    "50": {
      "hilbert": { "cond": ..., "gauss_err": ..., "svd_err": ..., "gs_err": ... },
      "spd":     { ... }
    },
    ...
  }
}
"""
from __future__ import annotations

import json
import math
import sys
import time
import warnings
from pathlib import Path

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_PART3 = Path(__file__).resolve().parent
_ROOT = _PART3.parent
for _p in [str(_ROOT), str(_ROOT / "part1"), str(_ROOT / "part2")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Force UTF-8 output on Windows (tránh UnicodeEncodeError với CP1252)
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from data_gen import (
    calculate_condition_number,
    generate_hilbert_matrix,
    generate_spd_matrix,
)
from solvers import solve_system

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PERF_SIZES      = [50, 100, 200, 500, 1000]
STABILITY_SIZES = [50, 100, 200, 500]
METHODS         = ["gauss", "svd", "gauss_seidel"]
N_RUNS          = 5   # số lần lặp để lấy thời gian trung bình


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _relative_error_np(A_list, x_list, b_list) -> float:
    """Tính ||Ax - b||₂ / ||b||₂ bằng NumPy (tránh overflow)."""
    import numpy as np
    A = np.array(A_list, dtype=float)
    x = np.array(x_list, dtype=float)
    b = np.array(b_list, dtype=float)
    residual = A @ x - b
    norm_b = float(np.linalg.norm(b))
    return float(np.linalg.norm(residual) / norm_b) if norm_b != 0.0 else float("inf")


def _avg_time_and_error(
    A: list[list[float]], b: list[float], method: str
) -> tuple[float, float]:
    """
    Gọi solve_system N_RUNS lần, trả về (avg_time_s, relative_error).
    Ném ngoại lệ nếu solver báo lỗi (được xử lý bởi caller).
    """
    total_time = 0.0
    x_last: list[float] = []

    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        x_last = solve_system(A, b, method)
        total_time += time.perf_counter() - t0

    avg_time = total_time / N_RUNS
    error = _relative_error_np(A, x_last, b)
    return avg_time, error


def _b_from_matrix(A: list[list[float]]) -> list[float]:
    """Sinh vector b = A @ ones (nghiệm x=ones, thuận tiện kiểm tra sai số)."""
    n = len(A)
    return [sum(A[i][j] for j in range(n)) for i in range(n)]


# ---------------------------------------------------------------------------
# Public benchmark functions
# ---------------------------------------------------------------------------

def run_performance_benchmark() -> dict:
    """
    Đo thời gian trung bình và sai số tương đối trên SPD matrix.
    sizes = PERF_SIZES = [50, 100, 200, 500, 1000]

    Returns dict phù hợp với key "performance" trong JSON output.
    """
    perf: dict[str, dict[str, dict]] = {m: {} for m in METHODS}

    for n in PERF_SIZES:
        print(f"  [PERF] n={n} ...", flush=True)
        A = generate_spd_matrix(n)
        b = _b_from_matrix(A)

        for method in METHODS:
            try:
                avg_t, err = _avg_time_and_error(A, b, method)
                perf[method][str(n)] = {"time_s": avg_t, "error": err}
            except Exception as e:
                perf[method][str(n)] = {"time_s": None, "error": str(e)}

    return perf


def run_stability_benchmark() -> dict:
    """
    So sánh độ ổn định: Hilbert (ill-conditioned) vs SPD (well-conditioned).
    sizes = STABILITY_SIZES = [50, 100, 200, 500]
    Chỉ lấy condition_number và sai số tương đối (KHÔNG đo thời gian).

    Returns dict phù hợp với key "stability" trong JSON output.
    """
    stability: dict[str, dict] = {}

    for n in STABILITY_SIZES:
        stability[str(n)] = {}

        for label, gen_fn in [("hilbert", generate_hilbert_matrix),
                               ("spd",     generate_spd_matrix)]:
            print(f"  [STAB] n={n} ({label}) ...", flush=True)
            A = gen_fn(n)
            b = _b_from_matrix(A)
            cond = calculate_condition_number(A)

            entry: dict[str, object] = {"cond": cond}

            # Gauss error
            try:
                x = solve_system(A, b, "gauss")
                entry["gauss_err"] = _relative_error_np(A, x, b)
            except Exception as e:
                entry["gauss_err"] = str(e)

            # SVD error
            try:
                x = solve_system(A, b, "svd")
                entry["svd_err"] = _relative_error_np(A, x, b)
            except Exception as e:
                entry["svd_err"] = str(e)

            # Gauss-Seidel error (chỉ hợp lệ với SPD; Hilbert không chéo trội)
            try:
                x = solve_system(A, b, "gauss_seidel")
                entry["gs_err"] = _relative_error_np(A, x, b)
            except Exception as e:
                entry["gs_err"] = str(e)

            stability[str(n)][label] = entry

    return stability


# ---------------------------------------------------------------------------
# Table formatters
# ---------------------------------------------------------------------------

def _fmt_float(v: object, precision: int = 3) -> str:
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return "INF"
        return f"{v:.{precision}e}"
    if v is None:
        return "N/A"
    return str(v)


def _print_performance_table(perf: dict) -> None:
    col_n   = 6
    col_t   = 14
    header  = (
        f"{'n':>{col_n}} | "
        f"{'Gauss (s)':>{col_t}} | "
        f"{'SVD (s)':>{col_t}} | "
        f"{'Gauss-Seidel (s)':>{col_t}}"
    )
    sep = "─" * len(header)

    print()
    print(sep)
    print(" BẢNG 1: THỜI GIAN TRUNG BÌNH (avg của 5 lần chạy) — Ma trận SPD")
    print(sep)
    print(header)
    print(sep)

    for n in PERF_SIZES:
        key = str(n)
        g   = perf["gauss"].get(key, {})
        s   = perf["svd"].get(key, {})
        gs  = perf["gauss_seidel"].get(key, {})

        t_g  = _fmt_float(g.get("time_s"))
        t_s  = _fmt_float(s.get("time_s"))
        t_gs = _fmt_float(gs.get("time_s"))

        print(
            f"{n:>{col_n}} | "
            f"{t_g:>{col_t}} | "
            f"{t_s:>{col_t}} | "
            f"{t_gs:>{col_t}}"
        )

    print(sep)
    print()


def _print_stability_table(stability: dict) -> None:
    col_n    = 5
    col_type = 8
    col_cond = 14
    col_err  = 14

    header = (
        f"{'n':>{col_n}} | "
        f"{'Loại':>{col_type}} | "
        f"{'κ₂(A)':>{col_cond}} | "
        f"{'Err_Gauss':>{col_err}} | "
        f"{'Err_SVD':>{col_err}} | "
        f"{'Err_GS':>{col_err}}"
    )
    sep = "─" * len(header)

    print(sep)
    print(" BẢNG 2: ĐỘ ỔN ĐỊNH — SAI SỐ TƯƠNG ĐỐI ||Ax-b||/||b||")
    print(sep)
    print(header)
    print(sep)

    for n in STABILITY_SIZES:
        key = str(n)
        for label in ["hilbert", "spd"]:
            entry = stability.get(key, {}).get(label, {})
            cond     = _fmt_float(entry.get("cond"))
            g_err    = _fmt_float(entry.get("gauss_err"))
            s_err    = _fmt_float(entry.get("svd_err"))
            gs_err_v = entry.get("gs_err")

            # Gauss-Seidel không hội tụ với Hilbert → in ERR ngắn gọn
            if isinstance(gs_err_v, str):
                gs_err = "ERR"
            else:
                gs_err = _fmt_float(gs_err_v)

            print(
                f"{n:>{col_n}} | "
                f"{label.capitalize():>{col_type}} | "
                f"{cond:>{col_cond}} | "
                f"{g_err:>{col_err}} | "
                f"{s_err:>{col_err}} | "
                f"{gs_err:>{col_err}}"
            )

    print(sep)
    print(" ERR (GS/Hilbert): Ma trận Hilbert không chéo trội → Gauss-Seidel không hội tụ.")
    print()

def format_error_table(rows: list[dict]) -> str:
    header = (f"{'n':>5} | {'Loại':>8} | "
              f"{'Gauss (Err)':>15} | {'SVD (Err)':>15} | {'Gauss-Seidel (Err)':>18}")
    sep = "─" * len(header)
    lines = [
        "",
        " BẢNG SAI SỐ TƯƠNG ĐỐI — Đánh giá Độ Ổn Định Số Học (Stability)",
        sep,
        header,
        sep
    ]
    for r in rows:
        # Format số thực dạng khoa học (vd: 1.23e-15)
        def _fmt(val):
            if isinstance(val, float): return f"{val:.2e}"
            return str(val)
            
        lines.append(
            f"{r['n']:>5} | {r['type']:>8} | "
            f"{_fmt(r.get('gauss_err', 'N/A')):>15} | "
            f"{_fmt(r.get('svd_err', 'N/A')):>15} | "
            f"{_fmt(r.get('gs_err', 'N/A')):>18}"
        )
        # Thêm vạch kẻ ngang sau mỗi cụm n để dễ nhìn như bảng thời gian
        if r['type'] == 'Hilbert':
            lines.append(sep)
            
    lines.append(" Cột Gauss/Hilbert: Sai số bùng nổ (10^4 -> 10^10) chứng minh ma trận điều kiện kém.")
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    print("=" * 60)
    print(" BENCHMARK PHẦN 3: GIẢI HỆ PHƯƠNG TRÌNH & PHÂN TÍCH HIỆU NĂNG")
    print("=" * 60)

    print("\n[1/2] Đang chạy Performance Benchmark...")
    perf_data = run_performance_benchmark()

    print("\n[2/2] Đang chạy Stability Benchmark...")
    stab_data = run_stability_benchmark()

    # ── Lưu JSON ─────────────────────────────────────────────────────────
    results = {"performance": perf_data, "stability": stab_data}
    out_path = _PART3 / "benchmark_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Đã lưu kết quả vào: {out_path}\n")

    # ── In bảng ra terminal ───────────────────────────────────────────────
    _print_performance_table(perf_data)
    _print_stability_table(stab_data)