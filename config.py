# Hằng số epsilon để xử lý sai số số thực
EPSILON = 1e-12

def is_zero(x):
    """
    Kiểm tra một số có xấp xỉ bằng 0 hay không.
    """
    return abs(x) < EPSILON

def zero_rectify(value):
    """
    Khử giá trị về 0 nếu nó đủ nhỏ.
    """
    return 0.0 if is_zero(value) else value

import os
# Kích hoạt ANSI escape character trên Windows Terminal
os.system('') 

class TestLogger:
    # Bảng màu ANSI
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

    @classmethod
    def print_suite_header(cls, suite_name: str):
        print(f"\n{cls.CYAN}{cls.BOLD}={'='*75}{cls.RESET}")
        print(f"{cls.CYAN}{cls.BOLD}TEST SUITE: {suite_name.upper()}{cls.RESET}")
        print(f"{cls.CYAN}{cls.BOLD}={'='*75}{cls.RESET}")

    @classmethod
    def print_result(cls, test_name: str, passed: bool, details: str = ""):
        name_padded = f"{test_name[:42] + '...' if len(test_name) > 45 else test_name:<45}"
        
        if passed:
            status = f"{cls.GREEN}{cls.BOLD}PASSED{cls.RESET}"
            detail_str = f"{cls.GREEN}{details}{cls.RESET}" if details else ""
        else:
            status = f"{cls.RED}{cls.BOLD}FAILED{cls.RESET}"
            detail_str = f"{cls.RED}{details}{cls.RESET}" if details else ""
            
        print(f" {name_padded} {status}  {detail_str}")

    @classmethod
    def print_summary(cls, passed_count: int, total_count: int):
        print(f"{cls.CYAN}{'-'*75}{cls.RESET}")
        if passed_count == total_count:
            print(f"{cls.GREEN}{cls.BOLD} TỔNG KẾT: {passed_count}/{total_count} PASSED{cls.RESET}\n")
        else:
            print(f"{cls.RED}{cls.BOLD} TỔNG KẾT: {passed_count}/{total_count} PASSED - CẦN KIỂM TRA LẠI!{cls.RESET}\n")