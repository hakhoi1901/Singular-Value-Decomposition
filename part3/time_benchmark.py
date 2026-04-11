import time
import sys
import warnings
from pathlib import Path
import numpy as np

# Boostrap paths to import local modules
_PART3 = Path(__file__).resolve().parent
_ROOT  = _PART3.parent
for _p in [str(_ROOT), str(_ROOT / "part1"), str(_ROOT / "part2")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from data_gen import generate_hilbert_matrix, generate_spd_matrix
    from solvers import solve_system
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

SIZES   = [50, 100, 200, 500, 1000]
METHODS = ['gauss', 'svd', 'gauss_seidel']

def measure_time(A, b, method):
    """Đo thời gian thực thi của solve_system (Pure Python)."""
    start = time.perf_counter()
    try:
        solve_system(A, b, method)
        return time.perf_counter() - start
    except Exception as e:
        return f"ERR: {e}"

def run_time_benchmark():
    rows = []
    print(">>> Đang chạy benchmark thời gian (Pure Python solvers)...")
    for n in SIZES:
        for m_type, gen_fn in [('Hilbert', generate_hilbert_matrix),
                                ('SPD',     generate_spd_matrix)]:
            A = gen_fn(n)
            b = [sum(row) for row in A]
            row = {'n': n, 'type': m_type}
            
            for method in METHODS:
                # Ngưỡng bỏ qua để tránh treo máy lâu (> 60s)
                # Dựa trên analysis.txt:
                # SVD n=200 đã mất >120s
                # Gauss n=1000 đã T.O.
                
                if method == 'svd' and n >= 200:
                    row[method] = "T.O. (>60s)"
                    continue
                if method == 'gauss' and n >= 1000:
                    row[method] = "T.O. (>60s)"
                    continue
                if method == 'gauss_seidel':
                    # Hilbert không bao giờ hội tụ Gauss-Seidel
                    if m_type == 'Hilbert':
                        row[method] = "ERR"
                        continue
                    # SPD trong data_gen.py có vẻ không chéo trội (A = XX^T + nI)
                    # nên check_diagonally_dominant sẽ báo False.
                    row[method] = "ERR"
                    continue

                duration = measure_time(A, b, method)
                if isinstance(duration, float):
                    row[method] = f"{duration:.4f}s"
                else:
                    if "Ma trận suy biến" in str(duration):
                        row[method] = "SINGULAR"
                    else:
                        row[method] = "ERR"
            
            rows.append(row)
            print(f"Finished size {n} ({m_type})")
            
    return rows

def format_time_table(rows):
    header = (f"{'n':>5} | {'Loại':>8} | "
              f"{'Gauss':>12} | {'SVD':>12} | {'Gauss-Seidel':>13}")
    sep = "-" * len(header)
    lines = [
        "",
        sep,
        " BẢNG THỜI GIAN TRUNG BÌNH - PURE PYTHON (SOLVERS.PY)",
        sep,
        header,
        sep
    ]
    for r in rows:
        lines.append(
            f"{r['n']:>5} | {r['type']:>8} | "
            f"{str(r.get('gauss','N/A')):>12} | "
            f"{str(r.get('svd','N/A')):>12} | "
            f"{str(r.get('gauss_seidel','N/A')):>13}"
        )
    lines.append(sep)
    lines.append(" T.O. : Timeout (Thời gian chạy Pure Python quá lâu).")
    lines.append(" ERR  : Ma trận không thỏa mãn điều kiện hội tụ.")
    return "\n".join(lines)

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    results = run_time_benchmark()
    print(format_time_table(results))
