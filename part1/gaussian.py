"""
Khử Gauss với partial pivoting cho hệ Ax = b với A ∈ R^(m×n).

Vô số nghiệm (khi rank(A) < n và hệ tương thích): nghiệm tổng quát có dạng
    x = x_0 + t_1 * v_1 + ... + t_{n-r} * v_{n-r}
trong đó x_0 là một nghiệm riêng, v_k là cơ sở không gian nghiệm N(A), r = rank(A),
t_k ∈ R là tham số tự do.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import is_zero, zero_rectify


def gaussian_eliminate(A: list[list[float]], b: list[float]):
    """
    Đưa [A | b] về dạng bậc thang dòng (REF) bằng khử Gauss có chọn phần tử chốt cột.

    A có kích thước m × n (m hàng, n cột). len(b) phải bằng m.

    Khi không tìm được pivot tại một cột (các phần tử dưới đều ~0): in
    ``Không có pivot tại cột k`` và bỏ qua cột đó (không raise).

    Nếu sau đó gọi ``back_substitution`` trên tam giác thu được mà gặp đường chéo ~0,
    hàm đó sẽ in ``Hệ không có nghiệm duy nhất`` và trả về ``[]``.

    Returns:
        U: ma trận hệ số sau biến đổi (m × n), dạng bậc thang dòng
        c: vector vế phải sau biến đổi (độ dài m)
        swap_count: số lần hoán đổi dòng
    """
    if not A:
        raise ValueError("Ma trận A rỗng")

    m = len(A)
    n = len(A[0])
    if any(len(row) != n for row in A):
        raise ValueError("Mọi hàng của A phải cùng số cột")

    if len(b) != m:
        raise ValueError("Vector b phải có độ dài bằng số hàng của A")

    M = [list(A[i]) + [float(b[i])] for i in range(m)]
    swap_count = 0
    pivot_row = 0

    for j in range(n):
        if pivot_row >= m:
            break

        p = pivot_row
        best = abs(M[pivot_row][j])
        for i in range(pivot_row + 1, m):
            v = abs(M[i][j])
            if v > best:
                best = v
                p = i

        pivot = M[p][j]

        if is_zero(pivot):
            print(f"Không có pivot tại cột {j + 1}")
            continue

        if abs(pivot) < 1e-8:
            warnings.warn(
                "Pivot is very small; the system may be ill-conditioned.",
                UserWarning,
                stacklevel=2,
            )

        if p != pivot_row:
            M[pivot_row], M[p] = M[p], M[pivot_row]
            swap_count += 1

        pivot = M[pivot_row][j]
        for i in range(pivot_row + 1, m):
            if is_zero(M[i][j]):
                continue
            lik = M[i][j] / pivot
            for k in range(j, n + 1):
                M[i][k] -= lik * M[pivot_row][k]
                M[i][k] = zero_rectify(M[i][k])

        pivot_row += 1

    U = [row[:n] for row in M]
    c = [row[n] for row in M]
    return U, c, swap_count


def _is_upper_triangular(U: list[list[float]], tol: float = 1e-9) -> bool:
    n = len(U)
    for i in range(1, n):
        for j in range(i):
            if abs(U[i][j]) > tol:
                return False
    return True


def test_gaussian_eliminate():
    import os
    import sys
    import warnings

    _parent = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if _parent not in sys.path:
        sys.path.insert(0, _parent)

    from back_substitution import back_substitution
    from verify_solution import verify_solution

    warnings.simplefilter("ignore", UserWarning)

    test_cases = [
        {
            "name": "Don vi 2x2",
            "A": [[1.0, 0.0], [0.0, 1.0]],
            "b": [3.0, -1.0],
            "expect_swaps": 0,
        },
        {
            "name": "He 2x2 (partial pivot doi dong 1 lan)",
            "A": [[1.0, 2.0], [3.0, 4.0]],
            "b": [0.0, -2.0],
            "expect_x": [-2.0, 1.0],
            "expect_swaps": 1,
        },
        {
            "name": "Hoan vi [[0,1],[1,0]]",
            "A": [[0.0, 1.0], [1.0, 0.0]],
            "b": [2.0, 1.0],
            "expect_x": [1.0, 2.0],
            "expect_swaps": 1,
        },
        {
            "name": "3x3 cung ma tran test inverse",
            "A": [[1.0, 0.0, 5.0], [2.0, 1.0, 6.0], [3.0, 4.0, 0.0]],
            "b": [6.0, 10.0, 11.0],
            "expect_x": [1.0, 2.0, 1.0],
        },
        {
            "name": "4x4 cheo troi (nghiem don gian)",
            "A": [
                [4.0, 1.0, 0.0, 0.0],
                [1.0, 4.0, 1.0, 0.0],
                [0.0, 1.0, 4.0, 1.0],
                [0.0, 0.0, 1.0, 4.0],
            ],
            "b": [5.0, 6.0, 6.0, 5.0],
            "expect_x": [1.0, 1.0, 1.0, 1.0],
        },
        {
            "name": "Cot dau tien gan 0 — can chon chot xa hon",
            "A": [[1e-12, 1.0], [1.0, 1.0]],
            "b": [3.0, 3.0],
        },
        {
            "name": "Suy bien [[1,2],[2,4]] + b nhat quan (in + khong nghiem duy nhat)",
            "A": [[1.0, 2.0], [2.0, 4.0]],
            "b": [1.0, 2.0],
            "expect_non_unique": True,
        },
        {
            "name": "He 2x3 (nhieu an hon phuong trinh — co the vo so nghiem)",
            "A": [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
            "b": [3.0, 3.0],
            "skip_backsub": True,
        },
        {
            "name": "b sai kich thuoc",
            "A": [[1.0, 0.0], [0.0, 1.0]],
            "b": [1.0],
            "should_raise": ValueError,
        },
    ]

    for case in test_cases:
        print(f"[gaussian_eliminate] {case['name']}")
        try:
            U, c, swaps = gaussian_eliminate(case["A"], case["b"])
            if case.get("should_raise"):
                assert False, "expected ValueError"
            if not case.get("skip_backsub"):
                assert _is_upper_triangular(U), "U phai tam giac tren (he vuong day du hang)"
            if "expect_swaps" in case:
                assert swaps == case["expect_swaps"], f"swap_count={swaps}"
            if case.get("skip_backsub"):
                print("  PASSED")
                continue
            if case.get("expect_non_unique"):
                x = back_substitution(U, c)
                assert x == []
                print("  PASSED")
                continue
            x = back_substitution(U, c)
            if "expect_x" in case:
                for i, v in enumerate(case["expect_x"]):
                    assert abs(x[i] - v) < 1e-8, f"x[{i}]={x[i]}"
            e = verify_solution(case["A"], x, case["b"])
            assert e < 1e-8, f"||b-Ax||={e}"
            print("  PASSED")
        except ValueError as err:
            if case.get("should_raise") == ValueError:
                print(f"  PASSED ({err})")
            else:
                raise


if __name__ == "__main__":
    test_gaussian_eliminate()
