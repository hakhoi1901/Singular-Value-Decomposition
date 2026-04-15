import math
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from config import zero_rectify, TestLogger
try:
    from part1.test_cases import VERIFY_SOLUTION_TEST_CASES
except ImportError:
    from test_cases import VERIFY_SOLUTION_TEST_CASES
from utils import matvec, vector_norm


def verify_solution(A: list[list[float]], x: list[float], b: list[float]) -> float:
    """
    Tính sai số Euclidean (L2 norm) của vector sai số (residual) cho hệ phương trình tuyến tính Ax = b.

    Thuật toán:
        1. Tính tích Ax (ma trận A nhân vector x).
        2. Tính vector sai số r = b - Ax.
        3. Tính chuẩn L2 của vector sai số: e = sqrt(sum(ri^2)).

    Tham số:
        A: Ma trận hệ số (m × n)
        x: Vector nghiệm (n × 1)
        b: Vector vế phải (m × 1)

    Trả về:
        e: Sai số Euclidean (float)

    Xử lý ngoại lệ:
        - Nếu A hoặc x rỗng, trả về 0.0
    """
    if not A or not A[0] or not x:
        return 0.0

    # 1. Tinh tich Ax
    Ax = matvec(A, x)

    # 2. Tinh vector du r = b - Ax
    residual_vector = [bi - axi for bi, axi in zip(b, Ax)]

    # 3. Tinh chuan Euclidean (L2 Norm) cua vector du
    return vector_norm(residual_vector)


def test_verify_solution(test_cases: list[dict]): 
    import numpy as np
    TestLogger.print_suite_header("Kiểm Chứng Nghiệm (Verify Solution)")
    
    passed_count = 0
    total_count = len(test_cases)

    for idx, case in enumerate(test_cases):
        e = verify_solution(case['A'], case['x'], case['b'])

        try:
            A_np = np.array(case['A'], dtype=float)
            b_np = np.array(case['b'], dtype=float)
            x_given = np.array(case['x'], dtype=float)

            if case.get("use_lstsq"):
                x_np, _, _, _ = np.linalg.lstsq(A_np, b_np, rcond=None)
                Ax_given = A_np @ x_given
                assert np.allclose(Ax_given, b_np, atol=1e-9), f"A*x_given = {np.round(Ax_given,6)} != b = {b_np}"
                assert e < 1e-9, f"e = {e:.2e} (nen xap xi 0)"
                TestLogger.print_result(case['name'], True, f"(A*x_given ~ b [OK via lstsq], e = {e:.2e})")
                passed_count += 1

            elif case.get("expect_mismatch"):
                x_np = np.linalg.solve(A_np, b_np)
                is_different = not np.allclose(x_np, x_given, atol=1e-9)
                assert is_different, "x_np bat ngo khop voi x sai!"
                assert e > 0.5, f"e = {e:.2e} qua nho voi nghiem sai"
                TestLogger.print_result(case['name'], True, f"(x sai => x_np != x_given [OK], e = {e:.2e})")
                passed_count += 1

            elif case.get("expect_instability"):
                x_np = np.linalg.solve(A_np, b_np)
                is_drifted = not np.allclose(x_np, x_given, atol=1e-4)
                assert is_drifted, "Bất ngờ chưa, nghiệm không bị trôi dạt!"
                TestLogger.print_result(case['name'], True, f"(Bị nhiễu số học: x_np = {np.round(x_np, 2)} khác xa x_given, e = {e:.2e})")
                passed_count += 1

            else:
                x_np = np.linalg.solve(A_np, b_np)
                assert np.allclose(x_np, x_given, atol=1e-9), f"x_np = {np.round(x_np,6)} != x_given = {x_given}"
                TestLogger.print_result(case['name'], True, f"(x_np ~ x_given [OK], e = {e:.2e})")
                passed_count += 1

        except AssertionError as err:
            TestLogger.print_result(case['name'], False, f"(AssertionError: {err})")
        except np.linalg.LinAlgError as ex:
            TestLogger.print_result(case['name'], False, f"(LinAlgError: {ex})")
        except Exception as ex:
            TestLogger.print_result(case['name'], False, f"(Lỗi: {ex})")

    TestLogger.print_summary(passed_count, total_count)
    if passed_count < total_count:
        raise AssertionError(f"Co {total_count - passed_count} test that bai.")

if __name__ == "__main__":
    test_verify_solution(VERIFY_SOLUTION_TEST_CASES)
