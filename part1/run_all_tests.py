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


def run_tests():
    print("TESTING")

    print("\n=== [Gaussian eliminate] ===")
    test_gaussian_eliminate()

    print("\n=== [Back substitution] ===")
    test_back_substitution()

    print("\n=== [Determinant] ===")
    test_determinant()

    print("\n=== [Testing Verify Solution] ===")
    test_verify_solution()

    print("\n=== [Testing Rank and Basis] ===")
    test_rank_and_basis()

    print("\n=== [Testing Inverse] ===")
    test_inverse()

if __name__ == "__main__":
    run_tests()
