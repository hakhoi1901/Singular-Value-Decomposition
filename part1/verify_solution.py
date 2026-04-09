import math
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from config import zero_rectify


def verify_solution(A: list[list[float]], x: list[float], b: list[float]) -> float:
    n_rows = len(A)
    n_cols = len(A[0])

    # 1. Tinh tich Ax (Ma tran A nhan vector x)
    # Ket qua Ax la mot vector cot co n_rows phan tu
    Ax = []
    for i in range(n_rows):
        row_sum = 0.0
        for j in range(n_cols):
            row_sum += A[i][j] * x[j]
        Ax.append(row_sum)

    # 2. Tinh vector du r = b - Ax
    residual_vector = []
    for i in range(n_rows):
        diff = b[i] - Ax[i]
        residual_vector.append(diff)

    # 3. Tinh chuan Euclidean (L2 Norm) cua vector du: e = sqrt(sum(ri^2))
    sum_squares = sum(ri ** 2 for ri in residual_vector)
    e = math.sqrt(sum_squares)

    return float(e)


def test_verify_solution():
    import numpy as np

    # -------------------------------------------------------------------
    # Cac test case
    #   "solvable"      : He vuong, nghiem duy nhat -> dung np.linalg.solve
    #   "use_lstsq"     : He khong vuong             -> dung np.linalg.lstsq
    #   "expect_mismatch": x truyen vao SAI, nen x_np != x_given
    # -------------------------------------------------------------------
    test_cases = [
        {
            "name": "Nghiem chinh xac (Identity matrix)",
            "A": [[1, 0], [0, 1]], "x": [1.0, 2.0], "b": [1.0, 2.0],
        },
        {
            "name": "He 2x2 thong thuong",
            "A": [[1, 2], [3, 4]], "x": [-2.0, 1.0], "b": [0.0, -2.0],
        },
        {
            "name": "He 1x1 - nghiem xap xi (lam tron)",
            "A": [[3.0]], "x": [1/3], "b": [1.0],
        },
        {
            "name": "He 3x3 nguyen tam giac tren",
            "A": [[2, 1, -1], [0, 3, 2], [0, 0, 4]],
            "x": [3.0, -2.0, 3.0], 
            "b": [1.0, 0.0, 12.0],
        },
        {
            "name": "Nghiem SAI (x khong phai nghiem dung)",
            "A": [[2, 1], [5, 7]], "x": [10.0, 10.0], "b": [11.0, 13.0],
            "expect_mismatch": True,
        },
        {
            "name": "He 2x3 (lstsq - Ax~b, kiem tra qua e)",
            "A": [[1, 0, 0], [0, 1, 0]], "x": [3.0, 4.0, 0.0], "b": [3.0, 4.0],
            "use_lstsq": True,
        },
        {
            "name": "He 1x1 - Sai so phay dong (0.1 * 3 != 0.3)",
            "A": [[3.0]], "x": [0.1], "b": [0.3],
        },{
            "name": "Nhiễu nhị phân vô hạn (0.1 + 0.2 != 0.3)",
            "A": [[1.0, 1.0], [1.0, -1.0]], 
            "x": [0.1, 0.2], 
            "b": [0.3, -0.1],
        },
        {
            "name": "Chênh lệch hệ số khổng lồ (Scale Disparity)",
            "A": [[1e15, 1.0], [1.0, 1.0]], 
            "x": [1.0, 1.0], 
            "b": [1e15 + 1.0, 2.0],
        },
        {
            "name": "Hệ điều kiện kém (Near-Singular Matrix)",
            "A": [[1.0, 1.0], [1.0, 1.000000000000001]], 
            "x": [1.0, 1.0], 
            "b": [2.0, 2.000000000000001],
            "expect_instability": True
        },
        {
            "name": "Ma trận Hilbert 3x3 (Vua bất ổn định)",
            "A": [[1.0, 1/2, 1/3], 
                  [1/2, 1/3, 1/4], 
                  [1/3, 1/4, 1/5]], 
            "x": [1.0, 1.0, 1.0], 
            "b": [11/6, 13/12, 47/60],
        }
    ]

    print("--- Test verify_solution ---")
    failed_count = 0  # <--- Dùng biến đếm số test rớt cho chuẩn

    for idx, case in enumerate(test_cases):
        e = verify_solution(case['A'], case['x'], case['b'])
        print(f"  - {case['name']}")

        try:
            A_np = np.array(case['A'], dtype=float)
            b_np = np.array(case['b'], dtype=float)
            x_given = np.array(case['x'], dtype=float)

            if case.get("use_lstsq"):
                x_np, _, _, _ = np.linalg.lstsq(A_np, b_np, rcond=None)
                Ax_given = A_np @ x_given
                assert np.allclose(Ax_given, b_np, atol=1e-9), f"A*x_given = {np.round(Ax_given,6)} != b = {b_np}"
                assert e < 1e-9, f"e = {e:.2e} (nen xap xi 0)"
                print(f"    => PASSED  (A*x_given ~ b [OK via lstsq], e = {e:.2e})")

            elif case.get("expect_mismatch"):
                x_np = np.linalg.solve(A_np, b_np)
                is_different = not np.allclose(x_np, x_given, atol=1e-9)
                assert is_different, "x_np bat ngo khop voi x sai!"
                assert e > 0.5, f"e = {e:.2e} qua nho voi nghiem sai"
                print(f"    => PASSED  (x sai => x_np != x_given [OK], e = {e:.2e})")

            # --- KHỐI LỆNH MỚI THÊM CHO HỆ ĐIỀU KIỆN KÉM ---
            elif case.get("expect_instability"):
                x_np = np.linalg.solve(A_np, b_np)
                # Bắt buộc nghiệm giải ra (x_np) phải KHÁC XA nghiệm lý thuyết (x_given) do bị nhiễu
                is_drifted = not np.allclose(x_np, x_given, atol=1e-4)
                assert is_drifted, "Bất ngờ chưa, nghiệm không bị trôi dạt!"
                print(f"    => PASSED  (Bị nhiễu số học: x_np = {np.round(x_np, 2)} khác xa x_given, e = {e:.2e})")
            # -----------------------------------------------

            else:
                x_np = np.linalg.solve(A_np, b_np)
                assert np.allclose(x_np, x_given, atol=1e-9), f"x_np = {np.round(x_np,6)} != x_given = {x_given}"
                print(f"    => PASSED  (x_np ~ x_given [OK], e = {e:.2e})")

        except AssertionError as err:
            print(f"    => FAILED: {err}")
            failed_count += 1
        except np.linalg.LinAlgError as ex:
            print(f"    => FAILED (LinAlgError): {ex}")
            failed_count += 1
        except Exception as ex:
            print(f"    => FAILED (Loi): {ex}")
            failed_count += 1

    if failed_count > 0:
        raise AssertionError(f"Co {failed_count} bai test that bai trong verify_solution.")

if __name__ == "__main__":
    test_verify_solution()