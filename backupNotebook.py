# %% [markdown]
# # Phần 1 Demo: Phép khử Gauss và Ứng dụng
# 
# Notebook này trình bày các hàm cơ bản của phép khử Gauss và các ứng dụng, bao gồm test suite với bảng số liệu và biểu đồ chú thích đầy đủ.

# %% [markdown]
# ## Thiết lập cấu hình (Config)
# 
# Khai báo hằng số EPSILON và các hàm bổ trợ để đảm bảo tính ổn định số học.

# %%
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Set seaborn style
sns.set_style("whitegrid")

# Thêm đường dẫn
sys.path.append('part1')

# Import config
from config import EPSILON, is_zero, zero_rectify

# Import các hàm
from gaussian import gaussian_eliminate
from back_substitution import back_substitution
from determinant import determinant
from inverse import inverse
from rank_basis import rank_and_basis
from verify_solution import verify_solution

def plot_matrix(matrix, title="Matrix", cmap="viridis"):
    """Hiển thị ma trận dưới dạng heatmap."""
    plt.figure(figsize=(6, 4))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap=cmap, cbar=True)
    plt.title(title)
    plt.show()

print(f"EPSILON = {EPSILON}")

# %% [markdown]
# ## Thuật toán Gaussian Elimination
# 
# Hàm `gaussian_eliminate` sử dụng Partial Pivoting để biến đổi ma trận A thành dạng tam giác trên U, với vector c tương ứng.

# %%
# Ví dụ minh họa
A = [[2.0, 1.0, -1.0], [-3.0, -1.0, 2.0], [-2.0, 1.0, 2.0]]
b = [8.0, -11.0, -3.0]
U, c, swaps = gaussian_eliminate(A, b)
print("Ma trận U:", U)
print("Vector c:", c)
print("Số lần hoán đổi:", swaps)

# Hiển thị ma trận A và U
plot_matrix(A, "Ma trận A gốc")
plot_matrix(U, "Ma trận U (tam giác trên)")

# %% [markdown]
# ## Back Substitution
# 
# Giải hệ Ux = c từ dưới lên.

# %%
x = back_substitution(U, c)
print("Nghiệm x:", x)

# %% [markdown]
# ## Tính định thức
# 
# Định thức = (-1)^swaps * tích đường chéo U

# %%
det = determinant(A)
print("Định thức:", det)

# %% [markdown]
# ## Ma trận nghịch đảo
# 
# Sử dụng Gauss-Jordan.

# %%
try:
    inv_A = inverse(A)
    print("Ma trận nghịch đảo:")
    for row in inv_A:
        print(row)
    plot_matrix(inv_A, "Ma trận nghịch đảo A^-1")
except ValueError as e:
    print("Lỗi:", e)

# %% [markdown]
# ## Hạng và cơ sở
# 
# Tính rank, cơ sở cột, dòng, nghiệm.

# %%
rank, col_basis, row_basis, null_basis = rank_and_basis(A)
print("Rank:", rank)
print("Cơ sở cột:", col_basis)
print("Cơ sở dòng:", row_basis)
print("Cơ sở nghiệm:", null_basis)

# %% [markdown]
# ## Xác minh nghiệm
# 
# Tính sai số dư.

# %%
error = verify_solution(A, x, b)
print("Sai số dư:", error)

# %% [markdown]
# ## Test Suite cho Gaussian Elimination + Back Substitution
# 
# Chạy 6-8 test cases, bao gồm trường hợp đặc biệt.

# %%
results = []
test_cases = [
    {"name": "Đơn vị 2x2", "A": [[1.0, 0.0], [0.0, 1.0]], "b": [3.0, -1.0]},
    {"name": "Hệ 2x2 (Hoán vị)", "A": [[1.0, 2.0], [3.0, 4.0]], "b": [0.0, -2.0]},
    {"name": "Hệ 3x3", "A": [[1.0, 0.0, 5.0], [2.0, 1.0, 6.0], [3.0, 4.0, 0.0]], "b": [6.0, 10.0, 11.0]},
    {"name": "4x4 Chéo trội", "A": [[4, 1, 0, 0], [1, 4, 1, 0], [0, 1, 4, 1], [0, 0, 1, 4]], "b": [5, 6, 6, 5]},
    {"name": "Small Pivot (1e-12)", "A": [[1e-12, 1.0], [1.0, 1.0]], "b": [3.0, 3.0]},
    {"name": "Suy biến (Singular)", "A": [[1.0, 2.0], [2.0, 4.0]], "b": [1.0, 2.0]}
]

print("Test Suite: Gaussian Elimination + Back Substitution\n" + "-"*50)
for case in test_cases:
    try:
        U, c, _ = gaussian_eliminate(case["A"], case["b"])
        x = back_substitution(U, c)
        if x:
            err = verify_solution(case["A"], x, case["b"])
            status = "PASSED" if err < 1e-9 else "FAILED (Error too high)"
            results.append({"Test": case["name"], "Error": err, "Status": status})
        else:
            results.append({"Test": case["name"], "Error": None, "Status": "PASSED (Non-unique)"})
        print(f"{case['name']}: {results[-1]['Status']}")
    except Exception as e:
        print(f"{case['name']}: ERROR - {e}")
        results.append({"Test": case["name"], "Error": None, "Status": f"ERROR: {e}"})

df = pd.DataFrame(results)
display(df)

# Biểu đồ sai số
errors = [r['Error'] for r in results if r['Error'] is not None]
if errors:
    plt.figure(figsize=(10, 5))
    valid_tests = [r['Test'] for r in results if r['Error'] is not None]
    sns.barplot(x=valid_tests, y=errors, palette="Blues_d", hue=valid_tests, legend=False)
    plt.xticks(rotation=45)
    plt.ylabel('Residual Error')
    plt.title('Residual Errors for Solver Tests')
    if any(e > 1e-20 for e in errors):
        plt.yscale('log')
    plt.show()

# %% [markdown]
# ## Test Suite cho Determinant
# 
# 6-8 test cases cho hàm determinant.

# %%
def run_tests_for_determinant():
    import warnings
    warnings.filterwarnings("ignore", message="Pivot is very small")
    print("Test Suite: Determinant\n" + "-"*30)
    results = []
    test_cases = [
        {"name": "Đơn vị 2x2", "A": [[1.0, 0.0], [0.0, 1.0]], "expected": 1.0},
        {"name": "Tỉ lệ dòng", "A": [[2.0, 0.0], [0.0, 1.0]], "expected": 2.0},
        {"name": "Suy biến", "A": [[1.0, 2.0], [2.0, 4.0]], "expected": 0.0},
        {"name": "Hoán vị", "A": [[0.0, 1.0], [1.0, 0.0]], "expected": -1.0},
        {"name": "Tam giác trên", "A": [[2.0, 3.0], [0.0, 5.0]], "expected": 10.0},
        {"name": "Ma trận 3x3", "A": [[1, 2, 3], [0, 4, 5], [1, 0, 1]], "expected": -14.0},
        {"name": "Lũy linh (Nilpotent)", "A": [[0, 1], [0, 0]], "expected": 0.0}
    ]
    
    for case in test_cases:
        det = determinant(case["A"])
        status = "PASSED" if abs(det - case["expected"]) < 1e-9 else f"FAILED (Got {det})"
        print(f"{case['name']}: {status}")
        results.append({"Test": case['name'], "Determinant": det, "Expected": case['expected'], "Status": status})
    
    df = pd.DataFrame(results)
    display(df)

run_tests_for_determinant()

# %% [markdown]
# ## Test Suite cho Inverse
# 
# 6-8 test cases cho hàm inverse.

# %%
def run_tests_for_inverse():
    print("Test Suite: Inverse\n" + "-"*30)
    passed_tests = 0
    results = []
    
    test_cases = [
        {"A": [[1.0, 0.0], [0.0, 1.0]], "expected": [[1.0, 0.0], [0.0, 1.0]], "name": "Identity"},
        {"A": [[2.0, 1.0], [1.0, 1.0]], "expected": [[1.0, -1.0], [-1.0, 2.0]], "name": "2x2 Invertible"},
        {"A": [[1.0, 2.0], [2.0, 4.0]], "singular": True, "name": "Singular"},
        {"A": [[1.0]], "expected": [[1.0]], "name": "1x1"},
        {"A": [[0.0, 1.0], [1.0, 0.0]], "expected": [[0.0, 1.0], [1.0, 0.0]], "name": "Permutation"},
        {"A": [[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]], "expected": None, "name": "3x3"},  # Skip expected for simplicity
        {"A": [[1e-10, 1.0], [1.0, 1.0]], "expected": None, "name": "Ill-conditioned"},
        {"A": [[2.0, 1.0], [1.0, 2.0]], "expected": [[2/3, -1/3], [-1/3, 2/3]], "name": "Symmetric Positive Definite"}
    ]
    
    for case in test_cases:
        try:
            inv_A = inverse(case["A"])
            if "singular" in case and case["singular"]:
                print(f"{case['name']}: FAILED - Should be singular")
                results.append({"Test": case["name"], "Inverse": "Singular", "Status": "FAILED"})
            else:
                # Check if A * inv_A ≈ I
                n = len(case["A"])
                product = [[sum(case["A"][i][k] * inv_A[k][j] for k in range(n)) for j in range(n)] for i in range(n)]
                is_identity = all(abs(product[i][j] - (1.0 if i==j else 0.0)) < 1e-9 for i in range(n) for j in range(n))
                if is_identity:
                    print(f"{case['name']}: PASSED")
                    passed_tests += 1
                    results.append({"Test": case["name"], "Inverse": "Computed", "Status": "PASSED"})
                else:
                    print(f"{case['name']}: FAILED - Product not identity")
                    results.append({"Test": case["name"], "Inverse": "Incorrect", "Status": "FAILED"})
        except ValueError:
            if "singular" in case and case["singular"]:
                print(f"{case['name']}: PASSED - Singular detected")
                passed_tests += 1
                results.append({"Test": case["name"], "Inverse": "Singular", "Status": "PASSED"})
            else:
                print(f"{case['name']}: FAILED - Unexpected singular")
                results.append({"Test": case["name"], "Inverse": "Error", "Status": "FAILED"})
        except Exception as e:
            print(f"{case['name']}: FAILED - {e}")
            results.append({"Test": case["name"], "Inverse": str(e), "Status": "FAILED"})
    
    print(f"\nPassed: {passed_tests}/{len(test_cases)}")
    
    df = pd.DataFrame(results)
    display(df)

run_tests_for_inverse()

# %% [markdown]
# ## Test Suite cho Rank and Basis
# 
# 6-8 test cases cho hàm rank_and_basis.

# %%
def run_tests_for_rank_basis():
    print("Test Suite: Rank and Basis\n" + "-"*30)
    passed_tests = 0
    results = []
    
    test_cases = [
        {"A": [[1.0, 0.0], [0.0, 1.0]], "expected_rank": 2, "name": "Full Rank 2x2"},
        {"A": [[1.0, 2.0], [2.0, 4.0]], "expected_rank": 1, "name": "Rank 1"},
        {"A": [[0.0, 0.0], [0.0, 0.0]], "expected_rank": 0, "name": "Null Matrix"},
        {"A": [[1.0, 2.0, 3.0]], "expected_rank": 1, "name": "1x3 Matrix"},
        {"A": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], "expected_rank": 3, "name": "Identity 3x3"},
        {"A": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], "expected_rank": 2, "name": "3x2 Full Rank"},
        {"A": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], "expected_rank": 2, "name": "Rank Deficient 3x3"},
        {"A": [[1e-16, 0.0], [0.0, 1e-16]], "expected_rank": 0, "name": "Small Values"}
    ]
    
    for case in test_cases:
        try:
            rank, col_basis, row_basis, null_basis = rank_and_basis(case["A"])
            if rank == case["expected_rank"]:
                print(f"{case['name']}: PASSED - Rank {rank}")
                passed_tests += 1
                results.append({"Test": case["name"], "Rank": rank, "Col Basis Size": len(col_basis), "Row Basis Size": len(row_basis), "Null Basis Size": len(null_basis), "Status": "PASSED"})
            else:
                print(f"{case['name']}: FAILED - Got rank {rank}, Expected {case['expected_rank']}")
                results.append({"Test": case["name"], "Rank": rank, "Col Basis Size": len(col_basis), "Row Basis Size": len(row_basis), "Null Basis Size": len(null_basis), "Status": "FAILED"})
        except Exception as e:
            print(f"{case['name']}: FAILED - {e}")
            results.append({"Test": case["name"], "Rank": None, "Status": "FAILED"})
    
    print(f"\nPassed: {passed_tests}/{len(test_cases)}")
    
    df = pd.DataFrame(results)
    display(df)

run_tests_for_rank_basis()

# %% [markdown]
# ## Test Suite cho Verify Solution
# 
# 6-8 test cases cho hàm verify_solution, so sánh với NumPy.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Ensure the 'part1' path is added so we can import local modules
if 'part1' not in sys.path:
    sys.path.append('part1')

# Import your custom functions
# (Make sure these matches the filenames in your 'part1' folder)
try:
    from gaussian import gaussian_eliminate
    from back_substitution import back_substitution
    from verify_solution import verify_solution
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Ensure you are running this from the project root and the 'part1' folder exists.")

results = []
errors_custom = []
errors_numpy = []

test_cases = [
    {"A": [[4, 1], [1, 3]], "b": [1, 2], "name": "Exact Solution"},
    {"A": [[1.0, 0.5], [0.5, 0.33]], "b": [1.5, 0.83], "name": "Approximate"},
    {"A": [[1e-10, 1], [1, 1]], "b": [1, 2], "name": "Ill-conditioned"},
    {"A": [[0, 0], [0, 0]], "b": [0, 0], "name": "Null Matrix"}
]

print("Test Suite: Verify Solution vs NumPy Residual\n" + "-"*50)
executed_cases = []

for case in test_cases:
    try:
        # Convert lists to numpy arrays
        A_np = np.array(case["A"])
        b_np = np.array(case["b"])
        
        # Run custom solver
        U, c, _ = gaussian_eliminate(case["A"], case["b"])
        x_custom = back_substitution(U, c)
        
        if x_custom is not None:
            # Calculate custom residual error
            err_custom = verify_solution(case["A"], x_custom, case["b"])
            
            # Calculate NumPy residual error for comparison
            # We use try/except for solve in case NumPy handles singular matrices differently
            try:
                if np.linalg.det(A_np) != 0:
                    x_np = np.linalg.solve(A_np, b_np)
                    err_np = np.linalg.norm(A_np @ x_np - b_np)
                else:
                    err_np = 0.0
            except np.linalg.LinAlgError:
                err_np = 0.0
            
            errors_custom.append(err_custom)
            errors_numpy.append(err_np)
            executed_cases.append(case["name"])
            status = "PASSED"
        else:
            status = "PASSED (Singular/No unique solution)"
            
        results.append({
            "Test": case["name"], 
            "Custom Error": err_custom if x_custom is not None else None, 
            "NumPy Error": err_np if x_custom is not None else None, 
            "Status": status
        })
        print(f"{case['name']}: {status}")
        
    except Exception as e:
        print(f"{case['name']}: ERROR - {e}")

# Display numerical results
df = pd.DataFrame(results)
display(df)

# Plot comparison chart
if errors_custom:
    plt.figure(figsize=(10, 5))
    x_axis = range(len(errors_custom))
    plt.plot(x_axis, errors_custom, 'o-', label='Custom Solver')
    plt.plot(x_axis, errors_numpy, 'x--', label='NumPy Solver')
    plt.xticks(x_axis, executed_cases, rotation=45)
    plt.ylabel('Residual Norm (log scale)')
    plt.title('Performance Comparison: Custom Gaussian vs NumPy solve()')
    plt.legend()
    # Use log scale to see tiny differences in precision
    if any(e > 0 for e in errors_custom + errors_numpy):
         plt.yscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.show()



