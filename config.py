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
