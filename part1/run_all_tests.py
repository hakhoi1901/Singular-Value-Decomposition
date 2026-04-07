# run_all_tests.py

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from verify_solution import test_verify_solution
from rank_basis import test_rank_and_basis
from inverse import test_inverse

def run_tests():
    print("TESTING")
    
    print("\n--- Testing Verify Solution ---")
    test_verify_solution()
    
    print("\n--- Testing Rank and Basis ---")
    test_rank_and_basis()
    
    print("\n--- Testing Inverse ---")
    test_inverse()

if __name__ == "__main__":
    run_tests()
