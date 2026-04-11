# ============================================================================
# Back Substitution Test Cases
# ============================================================================
BACK_SUBSTITUTION_TEST_CASES = [
    {
        "name": "2x2 nghiem nguyen",
        "U": [[1.0, 2.0], [0.0, 3.0]],
        "c": [7.0, 9.0],
        "expect_x": [1.0, 3.0],
    },
    {
        "name": "3x3 tam giac tren day du",
        "U": [[2.0, -1.0, 3.0], [0.0, 4.0, 1.0], [0.0, 0.0, 5.0]],
        "c": [9.0, 11.0, 15.0],
        "expect_x": [1.0, 2.0, 3.0],
    },
    {
        "name": "Don vi 4x4",
        "U": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "c": [1.0, -2.0, 3.0, -4.0],
        "expect_x": [1.0, -2.0, 3.0, -4.0],
    },
    {
        "name": "1x1",
        "U": [[8.0]],
        "c": [24.0],
        "expect_x": [3.0],
    },
    {
        "name": "Cheo lon (scale)",
        "U": [[1.0e6, 2.0], [0.0, 1.0e6]],
        "c": [-5.0e6 + 6.0, 3.0e6],
        "expect_x": [-5.0, 3.0],
    },
    {
        "name": "Duong cheo suy bien — tra ve rong",
        "U": [[1.0, 0.0], [0.0, 0.0]],
        "c": [1.0, 0.0],
        "expect_x": [],
    },
    {
        "name": "c sai do dai",
        "U": [[1.0, 0.0], [0.0, 1.0]],
        "c": [1.0],
        "should_raise": ValueError,
    },
    {
        "name": "U khong vuong (hang lech)",
        "U": [[1.0, 2.0, 3.0]],
        "c": [1.0],
        "should_raise": ValueError,
    },
    {
        "name": "Ma tran rong",
        "U": [],
        "c": [],
        "expect_x": [],
    },
]

# ============================================================================
# Determinant Test Cases
# ============================================================================
DETERMINANT_TEST_CASES = [
        {
            "name": "Don vi 2x2",
            "A": [[1.0, 0.0], [0.0, 1.0]],
            "expected": 1.0,
        },
        {
            "name": "2x2 [[1,2],[3,4]]",
            "A": [[1.0, 2.0], [3.0, 4.0]],
            "expected": -2.0,
        },
        {
            "name": "2x2 [[2,1],[1,3]] (det=5)",
            "A": [[2.0, 1.0], [1.0, 3.0]],
            "expected": 5.0,
        },
        {
            "name": "Hoan vi [[0,1],[1,0]]",
            "A": [[0.0, 1.0], [1.0, 0.0]],
            "expected": -1.0,
        },
        {
            "name": "Suy bien [[1,2],[2,4]]",
            "A": [[1.0, 2.0], [2.0, 4.0]],
            "expected": 0.0,
        },
        {
            "name": "Tam giac tren — tich duong cheo",
            "A": [[2.0, 5.0, 1.0], [0.0, 3.0, 4.0], [0.0, 0.0, 6.0]],
            "expected": 36.0,
        },
        {
            "name": "3x3 inverse test",
            "A": [[1.0, 0.0, 5.0], [2.0, 1.0, 6.0], [3.0, 4.0, 0.0]],
            "expected": 1.0,
        },
        {
            "name": "Cheo diag(1,2,3,4)",
            "A": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0, 0.0],
                [0.0, 0.0, 3.0, 0.0],
                [0.0, 0.0, 0.0, 4.0],
            ],
            "expected": 24.0,
        },
        {
            "name": "1x1",
            "A": [[-3.5]],
            "expected": -3.5,
        },
        {
            "name": "Toan khong 3x3",
            "A": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            "expected": 0.0,
        },
        {
            "name": "Hang lap — rank 1",
            "A": [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
            "expected": 0.0,
        },
        {
            "name": "Ma trận 3x3 (Đã sửa lại Toán: = 2.0)",
            "A": [[1, 2, 3], [0, 4, 5], [1, 0, 1]],
            "expected": 2.0,
        },
        {
            "name": "Khong vuong",
            "A": [[1.0, 2.0, 3.0]],
            "should_raise": ValueError,
        },
    ]

# ============================================================================
# Gaussian Eliminate Test Cases
# ============================================================================
GAUSSIAN_ELIMINATE_TEST_CASES = [
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
            "name": "Suy bien [[1,2],[2,4]] + b nhat quan",
            "A": [[1.0, 2.0], [2.0, 4.0]],
            "b": [1.0, 2.0],
            "expect_non_unique": True,
        },
        {
            "name": "He vo nghiem: [[1,1],[1,1]] + b bat nhat quan",
            "A": [[1.0, 1.0], [1.0, 1.0]],
            "b": [1.0, 2.0],
            "expect_non_unique": True,  # back_substitution tra ve [] vi dong 0 = non-zero
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

# ============================================================================
# Inverse Test Cases
# ============================================================================
INVERSE_TEST_CASES = [
        {
            "name": "Ma trận đơn vị 2x2",
            "input": [[1.0, 0.0], [0.0, 1.0]]
        },
        {
            "name": "Ma trận số âm",
            "input": [[-1.0, -2.0], [-3.0, -4.0]]
        },
        {
            "name": "Ma trận suy biến (Hàng 2 gấp đôi hàng 1)",
            "input": [[1.0, 2.0], [2.0, 4.0]],
            "should_raise": ValueError
        },
        {
            "name": "Ma trận 3x3",
            "input": [[1.0, 0.0, 5.0], [2.0, 1.0, 6.0], [3.0, 4.0, 0.0]]
        },
        {
            "name": "Ma trận có phần tử chốt bằng 0 ở giữa",
            "input": [[0.0, 1.0], [1.0, 0.0]]
        },
        {
            "name": "Ma trận số thực nhỏ (Điều kiện kém / Ill-conditioned)",
            "input": [[1.0, 2.0], [1.0, 2.00000000001]]
        },
        {
            "name": "Ma trận suy biến do số cực nhỏ (< EPSILON)",
            "input": [[1.0, 2.0], [1.0, 2.0 + 1e-16]],
            "should_raise": ValueError
        },
        {
            "name": "Ma trận không vuông 2x3 (kiểm tra dimension check)",
            "input": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            "should_raise": ValueError
        }
    ]

# ============================================================================
# Rank and Basis Test Cases
# ============================================================================
RANK_BASIS_TEST_CASES = [
        {
            "name": "Ma trận đơn vị 3x3 (Full Rank)",
            "input": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "exp_rank": 3,
            "null_is_empty": True
        },
        {
            "name": "Ma trận không 2x3",
            "input": [[0, 0, 0], [0, 0, 0]],
            "exp_rank": 0,
            "exp_null_dim": 3
        },
        {
            "name": "Ma trận có dòng phụ thuộc tuyến tính (R3 = R1 + R2)",
            "input": [[1, 2, 1], [0, 1, 1], [1, 3, 2]],
            "exp_rank": 2,
            "exp_null_dim": 1
        },
        {
            "name": "Ma trận 1 dòng nhiều cột",
            "input": [[1, 2, 3, 4]],
            "exp_rank": 1,
            "exp_null_dim": 3
        },
        {
            "name": "Ma trận có cột toàn số 0 ở giữa",
            "input": [[1, 0, 2], [3, 0, 4]],
            "exp_rank": 2,
            "exp_null_dim": 1
        },
        {
            "name": "Ma trận vuông suy biến",
            "input": [[1, 1], [1, 1]],
            "exp_rank": 1,
            "exp_null_dim": 1
        },
        {
            "name": "Ma trận số thực cực nhỏ (< EPSILON)",
            "input": [[1, 2], [0, 1e-16]],
            "exp_rank": 1,
            "exp_null_dim": 1
        },
        {
            "name": "Ma trận kích thước 1x1",
            "input": [[5.0]],
            "exp_rank": 1,
            "exp_null_dim": 0
        }
    ]

# ============================================================================
# Verify Solution Test Cases
# ============================================================================
VERIFY_SOLUTION_TEST_CASES = [
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