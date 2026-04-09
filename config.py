# Hằng số epsilon để xử lý sai số số thực
import math

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

def calculate_relative_error(A: list, x_hat: list, b: list) -> float:
    """
    Tính sai số tương đối ||A*x_hat - b||_2 / ||b||_2.
    """
    n = len(b)
    # Tính residual r = A*x_hat - b
    residual = []
    for i in range(n):
        ax_i = sum(A[i][j] * x_hat[j] for j in range(len(x_hat)))
        residual.append(ax_i - b[i])
    norm_r = math.sqrt(sum(r * r for r in residual))
    norm_b = math.sqrt(sum(bi * bi for bi in b))
    if is_zero(norm_b):
        return 0.0 if is_zero(norm_r) else float('inf')
    return norm_r / norm_b
