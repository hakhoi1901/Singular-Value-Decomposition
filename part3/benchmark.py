from __future__ import annotations
import sys
import warnings
import numpy as np
from pathlib import Path

_PART3 = Path(__file__).resolve().parent
_ROOT  = _PART3.parent

for _p in [str(_ROOT), str(_ROOT / "part1"), str(_ROOT / "part2")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data_gen import generate_hilbert_matrix, generate_spd_matrix, calculate_condition_number
from config import calculate_relative_error
from solvers import solve_system

SIZES   = [50, 100, 200]
METHODS = ['gauss', 'svd', 'gauss_seidel']

NUMPY_THRESHOLD = 200
# Dùng numpy để tính các n lớn (500,1000), có thể sửa lại sau

def _solve(A, b, A_np, b_np, method: str, n: int):
    """
    n <= NUMPY_THRESHOLD : dùng solve_system (pure Python, để kiểm chứng)
    n  > NUMPY_THRESHOLD : dùng numpy làm verification cho Gauss và SVD.
                           Gauss-Seidel vẫn qua solve_system (fail nhanh).
    """
    if n <= NUMPY_THRESHOLD:
        x = solve_system(A, b, method)
        return calculate_relative_error(A, x, b)

    if method == 'gauss':
        x = np.linalg.solve(A_np, b_np)          # LU — tương đương Gauss
        r = A_np @ x - b_np
        return float(np.linalg.norm(r) / np.linalg.norm(b_np))

    if method == 'svd':
        x, _, _, _ = np.linalg.lstsq(A_np, b_np, rcond=None)  # SVD bên trong
        r = A_np @ x - b_np
        return float(np.linalg.norm(r) / np.linalg.norm(b_np))

    if method == 'gauss_seidel':
        # Vẫn dùng solve_system: fail nhanh nếu không chéo trội
        x = solve_system(A, b, method)
        return calculate_relative_error(A, x, b)

    raise ValueError(f"Unknown method: {method}")


def run_stability_analysis(sizes=SIZES) -> list[dict]:
    rows = []
    for n in sizes:
        for m_type, gen_fn in [('Hilbert', generate_hilbert_matrix),
                                ('SPD',     generate_spd_matrix)]:
            A    = gen_fn(n)
            b    = [sum(row) for row in A]
            A_np = np.array(A)
            b_np = np.array(b)
            row  = {'n': n, 'type': m_type, 'kappa': calculate_condition_number(A)}
            print(f"Checking n={n} ({m_type}) ...", flush=True)
            for method in METHODS:
                try:
                    row[method] = _solve(A, b, A_np, b_np, method, n)
                except Exception as e:
                    row[method] = f'ERR: {e}'
            rows.append(row)
    return rows


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.3e}"
    if isinstance(v, str) and v.startswith('ERR'):
        return 'ERR'
    return str(v)


def format_table(rows: list[dict]) -> str:
    header = (f"{'n':>5} | {'Loại':>8} | {'κ₂(A)':>12} | "
              f"{'Gauss':>12} | {'SVD':>12} | {'Gauss-Seidel':>13}")
    sep = "-" * len(header)
    lines = [sep, header, sep]
    for r in rows:
        lines.append(
            f"{r['n']:>5} | {r['type']:>8} | {_fmt(r['kappa']):>12} | "
            f"{_fmt(r.get('gauss','N/A')):>12} | "
            f"{_fmt(r.get('svd','N/A')):>12} | "
            f"{_fmt(r.get('gauss_seidel','N/A')):>13}"
        )
    lines.append(sep)
    lines.append(" Gauss/SVD (n > 15): dùng numpy verify (solve_system quá chậm với n lớn).")
    lines.append(" ERR Gauss-Seidel  : Ma trận không chéo trội → không hội tụ.")
    return "\n".join(lines)


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    rows = run_stability_analysis()
    print(format_table(rows))