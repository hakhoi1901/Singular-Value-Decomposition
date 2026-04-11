# run_all_tests.py

import os
import sys

_PART1 = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_PART1, ".."))
for _p in (_ROOT, _PART1):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gaussian import test_gaussian_eliminate
from back_substitution import test_back_substitution
from determinant import test_determinant
from verify_solution import test_verify_solution
from rank_basis import test_rank_and_basis
from inverse import test_inverse

from test_cases import (
    GAUSSIAN_ELIMINATE_TEST_CASES,
    BACK_SUBSTITUTION_TEST_CASES,
    DETERMINANT_TEST_CASES,
    VERIFY_SOLUTION_TEST_CASES,
    RANK_BASIS_TEST_CASES,
    INVERSE_TEST_CASES
)


def run_tests():
    """
    Chạy tất cả các bài kiểm tra cho từng phần của đồ án.
    """
    print("TESTING")

    print("\n=== [Gaussian eliminate] ===")
    test_gaussian_eliminate(GAUSSIAN_ELIMINATE_TEST_CASES)

    print("\n=== [Back substitution] ===")
    test_back_substitution(BACK_SUBSTITUTION_TEST_CASES)

    print("\n=== [Determinant] ===")
    test_determinant(DETERMINANT_TEST_CASES)

    print("\n=== [Testing Rank and Basis] ===")
    test_rank_and_basis(RANK_BASIS_TEST_CASES)

    print("\n=== [Testing Inverse] ===")
    test_inverse(INVERSE_TEST_CASES)

    print("\n=== [Testing Verify Solution] ===")
    test_verify_solution(VERIFY_SOLUTION_TEST_CASES)

if __name__ == "__main__":
    run_tests()
