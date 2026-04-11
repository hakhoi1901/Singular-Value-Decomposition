# ============================================================================
# Decomposition Test Cases
# ============================================================================
DECOMPOSITION_TEST_CASES = [
        {
            "name": "Ma trận vuông full-rank 3x3",
            "input": [[4.0, 1.0, 2.0], [0.0, 3.0, -1.0], [2.0, 0.0, 1.0]],
            "tol": 1e-7,
        },
        {
            "name": "Ma trận chữ nhật ngang 2x3",
            "input": [[3.0, 1.0, 1.0], [-1.0, 3.0, 1.0]],
            "tol": 1e-7,
        },
        {
            "name": "Ma trận chữ nhật dọc 3x2",
            "input": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            "tol": 1e-7,
        },
        {
            "name": "Ma trận suy biến (hai dòng phụ thuộc)",
            "input": [[1.0, 2.0], [2.0, 4.0]],
            "tol": 1e-7,
        },
        {
            "name": "Ma trận toàn 0 kích thước 3x4",
            "input": [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
            "tol": 1e-8,
        },
        {
            "name": "Ma trận đơn vị 4x4",
            "input": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            "tol": 1e-8,
        },
        {
            "name": "Ma trận đường chéo có singular values lặp",
            "input": [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 1.0]],
            "tol": 1e-8,
        },
        {
            "name": "Ma trận chứa số âm",
            "input": [[-3.0, 2.0], [4.0, -1.0]],
            "tol": 1e-7,
        },
        {
            "name": "Ma trận kích thước 1x1",
            "input": [[-7.0]],
            "tol": 1e-8,
        },
        {
            "name": "Ma trận 1xN",
            "input": [[2.0, -1.0, 3.0, 0.5]],
            "tol": 1e-7,
        },
        {
            "name": "Ma trận Nx1",
            "input": [[2.0], [-1.0], [3.0], [0.5]],
            "tol": 1e-7,
        },
        {
            "name": "Ma trận số rất nhỏ gần EPSILON",
            "input": [[1e-13, 0.0], [0.0, 2e-13]],
            "tol": 1e-10,
        },
        {
            "name": "Ma trận Hilbert 3x3 (kiểm tra độ ổn định số học)",
            "input": [
                [1.0, 1/2, 1/3],
                [1/2, 1/3, 1/4],
                [1/3, 1/4, 1/5],
            ],
            "tol": 1e-6,
        },
        {
            "name": "Dữ liệu rỗng",
            "input": [],
            "should_raise": ValueError,
        },
        {
            "name": "Ma trận có 0 cột",
            "input": [[]],
            "should_raise": ValueError,
        },
        {
            "name": "Ma trận không cùng số cột",
            "input": [[1.0, 2.0], [3.0]],
            "should_raise": ValueError,
        },
    ]
    
# ============================================================================
# Diagonalization Test Cases
# ============================================================================
DIAGONALIZATION_TEST_CASES = [
        {
            "name": "Ma trận chéo 3x3, trị riêng phân biệt",
            "input": [[5.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, -1.0]],
            "tol": 1e-8,
        },
        {
            "name": "Ma trận tam giác trên có trị riêng thực phân biệt",
            "input": [[4.0, 1.0, 0.0], [0.0, 2.0, 1.0], [0.0, 0.0, -3.0]],
            "tol": 1e-6,
        },
        {
            "name": "Ma trận đối xứng có trị riêng lặp",
            "input": [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]],
            "tol": 1e-8,
        },
        {
            "name": "Ma trận đơn vị 4x4",
            "input": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            "tol": 1e-8,
        },
        {
            "name": "Ma trận suy biến nhưng chéo hóa được",
            "input": [[0.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]],
            "tol": 1e-8,
        },
        {
            "name": "Ma trận kích thước 1x1",
            "input": [[-7.0]],
            "tol": 1e-10,
        },
        {
            "name": "Khối Jordan bậc 2 (không chéo hóa được)",
            "input": [[2.0, 1.0], [0.0, 2.0]],
            "should_raise": ValueError,
        },
        {
            "name": "Ma trận quay có trị riêng phức",
            "input": [[0.0, -1.0], [1.0, 0.0]],
            "should_raise": ValueError,
        },
        {
            "name": "Ma trận rỗng",
            "input": [],
            "should_raise": ValueError,
        },
        {
            "name": "Ma trận không vuông",
            "input": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            "should_raise": ValueError,
        },
        {
            "name": "Ma trận không cùng số cột",
            "input": [[1.0, 2.0], [3.0]],
            "should_raise": ValueError,
        },
        {
            "name": "Ma trận tam giác dưới (kiểm tra khả năng quét và khử của QR)",
            "input": [[2.0, 0.0, 0.0], [1.0, 3.0, 0.0], [-1.0, 4.0, 5.0]],
            "tol": 1e-6,
        },
    ]