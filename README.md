# ĐỒ án 1 - Toán ứng dụng và thống kê

> **Môn học:** Toán ứng dụng và thống kê — Học kỳ 2, năm học 2025–2026  
> **GVHD:** ThS. Lê Nhựt Nam, ThS. Võ Nam Thục Đoan   
> **Ngôn ngữ:** Python 3.10+  

---

## Mục lục

- [Tổng quan dự án](#tổng-quan-dự-án)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Cơ sở lý thuyết](#cơ-sở-lý-thuyết)
- [Phần 1 — Khử Gauss và các ứng dụng](#phần-1--khử-gauss-và-các-ứng-dụng)
- [Phần 2 — Chéo hóa và Phân rã SVD](#phần-2--chéo-hóa-và-phân-rã-svd)
- [Phần 3 — Ứng dụng & Benchmark](#phần-3--ứng-dụng--benchmark)
- [Trực quan hóa Manim](#trực-quan-hóa-manim)
- [Cài đặt môi trường](#cài-đặt-môi-trường)
- [Chạy thử nghiệm](#chạy-thử-nghiệm)
- [Báo cáo](#báo-cáo)
- [Ghi chú kỹ thuật](#ghi-chú-kỹ-thuật)

---

## Tổng quan dự án

Dự án này triển khai toàn bộ pipeline tính toán **Singular Value Decomposition (SVD)** từ đầu, **không sử dụng bất kỳ hàm phân rã hoặc trị riêng có sẵn nào** từ NumPy, SciPy hay SymPy. Mọi thuật toán đều được xây dựng thủ công bằng Python thuần túy, dựa trên các phép toán đại số tuyến tính cơ bản.

Dự án được chia thành ba phần chính:

| Phần | Nội dung | Trạng thái |
|------|----------|-----------|
| **Part 1** | Khử Gauss, thế ngược, tính định thức, nghịch đảo, hạng, cơ sở | Hoàn thành |
| **Part 2** | Chéo hóa ma trận (`A = PDP⁻¹`) và Phân rã SVD (`A = UΣVᵀ`) | Hoàn thành |
| **Part 3** | Ứng dụng SVD, so sánh benchmark với NumPy | Hoàn thành |

---

## Cấu trúc thư mục

```
Singular Value Decomposition/
│
├── config.py                   # Hằng số EPSILON và tiện ích xử lý sai số số thực
│
├── part1/                      # Phần 1: Thuật toán nền tảng (Gaussian Elimination)
│   ├── back_substitution.py    # Thế ngược (back substitution) cho ma trận tam giác trên
│   ├── determinant.py          # Tính định thức ma trận
│   ├── gaussian.py             # Khử Gauss với partial pivoting
│   ├── inverse.py              # Tính ma trận nghịch đảo
│   ├── rank_basis.py           # Hạng, cơ sở cột, cơ sở hàng, không gian nghiệm
│   ├── run_all_tests.py        # Chạy toàn bộ bộ kiểm thử của Part 1
│   ├── verify_solution.py      # Xác minh nghiệm Ax = b
│   └── part1_demo.ipynb        # Notebook trình bày kết quả Part 1
│
├── part2/                      # Phần 2: Chéo hóa và SVD
│   ├── decomposition.py        # Thuật toán SVD (A = UΣVᵀ)
│   ├── diagonalization.py      # Chéo hóa ma trận (A = PDP⁻¹)
│   ├── manim_scene.py          # Scene Manim: trực quan hóa SVD
│   ├── manim_scene/            # Các scene Manim bổ sung
│   ├── demo_video.mp4          # Video demo đã render
│   ├── media/                  # Output render của Manim
│   └── README.md               # Hướng dẫn render Manim
│
├── part3/                      # Phần 3: Ứng dụng và Benchmark
│   ├── solvers.py              # Bộ giải sử dụng SVD
│   ├── benchmark.py            # So sánh hiệu năng với NumPy SVD
│   └── analysis.ipynb          # Notebook phân tích kết quả
│
├── report/                     # Báo cáo LaTeX
│   ├── report.tex              # Cấu trúc báo cáo
│   ├── content/                # Các chương nội dung
│   ├── assets/                 # Hình ảnh, biểu đồ trong báo cáo
│   └── report.pdf              # Phiên bản PDF thứ hai
│
├── config.py                   # Cấu hình toàn cục
└── requirements.txt            # Danh sách thư viện phụ thuộc
```

---

## Cơ sở lý thuyết

### Singular Value Decomposition

Với mọi ma trận thực $A \in \mathbb{R}^{m \times n}$, tồn tại phân rã:

$$A = U \Sigma V^T$$

trong đó:
- $U \in \mathbb{R}^{m \times m}$ — ma trận trực giao trái, các cột là **left singular vectors**
- $\Sigma \in \mathbb{R}^{m \times n}$ — ma trận đường chéo, với $\sigma_1 \geq \sigma_2 \geq \ldots \geq \sigma_r \geq 0$ là **singular values**
- $V^T \in \mathbb{R}^{n \times n}$ — chuyển vị của ma trận trực giao phải, các hàng là **right singular vectors**

> **Ghi chú:** Chi tiết về chứng minh Toán học, giải thuật và phân tích độ phức tạp ($O(n^3)$), vui lòng xem chi tiết tại [Báo cáo Kỹ thuật (PDF)](report/report.pdf).

---

## Phần 1 — Khử Gauss và các ứng dụng

### Mô tả

Cài đặt các thuật toán đại số tuyến tính cơ bản **không dùng thư viện ngoài**:

| Module | Hàm chính | Mô tả |
|--------|-----------|-------|
| `gaussian.py` | `gaussian_eliminate(A, b)` | Khử Gauss với partial pivoting cho hệ $Ax = b$ (ma trận $m \times n$) |
| `back_substitution.py` | `back_substitution(U, c)` | Thế ngược giải $Ux = c$ với $U$ tam giác trên |
| `determinant.py` | `determinant(A)` | Tính $\det(A)$ qua khử Gauss, tính $(-1)^{\text{swap}} \cdot \prod u_{ii}$ |
| `inverse.py` | `inverse(A)` | Tính $A^{-1}$ bằng phương pháp Gauss-Jordan |
| `rank_basis.py` | `rank_and_basis(A)` | Tính hạng, cơ sở cột, cơ sở hàng và không gian nghiệm |
| `verify_solution.py` | `verify_solution(A, x, b)` | Tính $\|b - Ax\|$ để xác minh nghiệm |

### Xử lý đặc biệt

- **Hệ vô nghiệm:** phát hiện qua dòng toàn zero với vế phải khác zero
- **Hệ vô số nghiệm:** in nghiệm tổng quát $x = x_0 + t_1 v_1 + \ldots + t_k v_k$
- **Ma trận hình chữ nhật ($m \times n$, $m \neq n$):** khử Gauss vẫn hoạt động bình thường, bỏ qua các cột không có pivot

### Chạy kiểm thử

```bash
cd part1
python run_all_tests.py
```

Hoặc từng module riêng lẻ:

```bash
python part1/gaussian.py
python part1/back_substitution.py
python part1/rank_basis.py
python part1/determinant.py
python part1/inverse.py
python part1/verify_solution.py
```

---

## Phần 2 — Chéo hóa và Phân rã SVD

### `decomposition.py` — SVD từ đầu

**API:**

```python
from part2.decomposition import decompose_svd

U, sigma, V_T = decompose_svd(A)
```

**Trả về:**
- `U` — `list[list[float]]`, kích thước $m \times m$, ma trận trực giao trái
- `sigma` — `list[float]`, độ dài $\min(m, n)$, singular values giảm dần, không âm
- `V_T` — `list[list[float]]`, kích thước $n \times n$, ma trận trực giao phải (đã chuyển vị)

**Ví dụ nhanh:**

```python
A = [[4.0, 1.0, 2.0],
     [0.0, 3.0, -1.0],
     [2.0, 0.0,  1.0]]

U, sigma, V_T = decompose_svd(A)
# Xác minh: U @ diag(sigma) @ V_T ≈ A
```

**Bộ kiểm thử tích hợp** (`python part2/decomposition.py`):

| Test case | Mô tả |
|-----------|-------|
| Ma trận vuông full-rank 3×3 | Trường hợp cơ bản |
| Ma trận hình chữ nhật 2×3, 3×2 | Tall & wide matrix |
| Ma trận suy biến (linearly dependent rows) | Singular values = 0 |
| Ma trận toàn số 0 | σᵢ = 0 với mọi i |
| Ma trận đơn vị 4×4 | σᵢ = 1 với mọi i |
| Singular values trùng lặp | Xử lý degeneracy |
| Ma trận số âm | Tổng quát hóa |
| Ma trận 1×1, 1×N, N×1 | Biên |
| Số rất nhỏ gần EPSILON | Độ ổn định số học |
| Input không hợp lệ | `ValueError` như mong đợi |

### `diagonalization.py` — Chéo hóa ma trận

**API:**

```python
from part2.diagonalization import diagonalize

P, D = diagonalize(A)
```

**Trả về:**
- `P` — `list[list[float]]`, kích thước $n \times n$, mỗi cột là một vector riêng
- `D` — `list[float]`, độ dài $n$, các trị riêng tương ứng với từng cột của $P$

**Ném `ValueError` khi:**
- Ma trận không vuông
- Ma trận không chéo hóa được trên $\mathbb{R}$ (có trị riêng phức hoặc thiếu vector riêng độc lập, ví dụ: khối Jordan)
- Sai số số học quá lớn sau khi xác minh

**Bộ kiểm thử tích hợp** (`python part2/diagonalization.py`):

| Test case | Kết quả kỳ vọng |
|-----------|-----------------|
| Ma trận chéo 3×3, trị riêng phân biệt | ✅ PASSED |
| Ma trận tam giác trên, trị riêng thực phân biệt | ✅ PASSED |
| Trị riêng lặp nhưng đủ vector riêng | ✅ PASSED |
| Ma trận đơn vị 4×4 | ✅ PASSED |
| Khối Jordan bậc 2 | ✅ Bắt đúng `ValueError` |
| Ma trận quay (trị riêng phức) | ✅ Bắt đúng `ValueError` |

### Chạy kiểm thử

```bash
python part2/decomposition.py
python part2/diagonalization.py
```

---

## Phần 3 — Ứng dụng & Benchmark

Phần này (đang phát triển) bao gồm:

- **`solvers.py`** — Bộ giải hệ phương trình tuyến tính và bài toán bình phương tối thiểu sử dụng SVD
- **`benchmark.py`** — So sánh hiệu năng và độ chính xác giữa cài đặt thủ công và NumPy SVD
- **`analysis.ipynb`** — Notebook Jupyter phân tích kết quả benchmark

---

## Trực quan hóa Manim

link video: 

Scene Manim trong `part2/manim_scene.py` trực quan hóa ý nghĩa hình học của SVD theo tiếp cận **"Rotate → Scale → Rotate"**.

### Nội dung video

Kịch bản trực quan hóa được chia thành 4 cảnh chính, dẫn dắt từ trực giác hình học đến thuật toán toán học chi tiết:

1. **Cảnh 1: Giải phẫu sự hỗn loạn (Anatomy of Chaos)**
   - Minh họa cách ma trận $A = \begin{bmatrix}3&1\\0&2\end{bmatrix}$ biến đường tròn đơn vị thành hình elip và làm mất đi góc vuông giữa hai vector cơ sở $\hat{i}, \hat{j}$. 
   - Đặt ra vấn đề: Cần SVD để khôi phục lại một hệ cơ sở trực giao thông qua trục chính của elip.

2. **Cảnh 2: Trực quan hóa hình học (Geometric Interpretation)**
   - Phân rã trực quan SVD từ phải sang trái: **Xoay ($V^T$) $\rightarrow$ Kéo giãn ($\Sigma$) $\rightarrow$ Xoay lại ($U$)**.
   - Chứng minh từng bước ma trận tác động lên không gian mà không làm mất tính trực giao.

3. **Cảnh 3: Thuật toán và Toán học phía sau SVD**
   - Chỉ ra cách tính toán $V^T$ qua trị riêng của $A^TA$, $\Sigma$ từ căn bậc hai của trị riêng, và $U$ từ công thức $u_i = \frac{A v_i}{\sigma_i}$.
   - Ghép nối các thành phần đại số vào đồ họa trực quan của elip.

4. **Cảnh 4: SVD vs Chéo hóa (Diagonalization)**
   - So sánh trực quan hình học của phép chéo hóa ($A = PDP^{-1}$) nhấn mạnh "tính bất biến" nằm trên vector riêng (không trực giao).
   - Nêu bật giới hạn của phương pháp chéo hóa (chỉ cho ma trận vuông thỏa mãn điệu kiện) so với khả năng đối phó mọi độ khó của SVD.

### Render video

```bash
# Xem trước nhanh (480p)
manim -pql part2/manim_scene.py Scene1_AnatomyOfChaos

# Chất lượng cao (1080p60)
manim -pqh part2/manim_scene.py Scene1_AnatomyOfChaos

# Xuất GIF
manim -pql --format=gif part2/manim_scene.py Scene1_AnatomyOfChaos
```

**Bảng flag Manim:**

| Flag | Ý nghĩa |
|------|---------|
| `-p` | Tự mở video sau khi render |
| `-ql` | Low quality (480p15) — nhanh |
| `-qm` | Medium quality (720p30) |
| `-qh` | High quality (1080p60) |
| `-s` | Chỉ xuất frame cuối (ảnh tĩnh) |
| `--format=gif` | Xuất định dạng GIF |

---

## Cài đặt môi trường

### Yêu cầu hệ thống

- Python 3.10 trở lên
- (Tùy chọn) LaTeX distribution (MiKTeX / TeX Live) để biên dịch báo cáo
- (Tùy chọn) FFmpeg để Manim render video

### Thiết lập

```bash
# 1. Clone repository
git clone https://github.com/hakhoi1901/Singular-Value-Decomposition.git
cd Singular-Value-Decomposition

# 2. Tạo môi trường ảo
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

# 3. Cài đặt thư viện
pip install -r requirements.txt
```

### Thư viện chính

| Thư viện | Mục đích |
|----------|---------|
| `manim` | Render animation trực quan hóa SVD |
| `numpy` | Xác minh kết quả và benchmark (KHÔNG dùng cho thuật toán lõi) |
| `jupyter` / `notebook` | Chạy các file `.ipynb` |
| `matplotlib` | Vẽ biểu đồ phân tích trong notebook |

> **Lưu ý quan trọng:** NumPy chỉ được phép sử dụng trong hai trường hợp:  
> (1) Xác minh/kiểm thử kết quả của thuật toán tự cài đặt,  
> (2) Dữ liệu đầu vào cho Manim để vẽ hình chính xác.  
> Tất cả thuật toán tính toán (SVD, trị riêng, v.v.) đều phải tự cài đặt bằng Python thuần.

---

## Chạy thử nghiệm

### Kiểm thử Part 1 

**Kiểm thử tất cả module**

```bash
python part1/run_all_tests.py
```

**Kiểm thử từng module**

```bash
# Khử Gauss
python part1/gaussian_elimination.py

# Thế ngược
python part1/back_substitution.py

# Tính định thức
python part1/determinant.py

# Tính nghịch đảo
python part1/inverse.py

# Tính hạng
python part1/rank.py

# Tính cơ sở
python part1/basis.py
```

### Kiểm thử Part 2

```bash
# SVD
python part2/decomposition.py

# Chéo hóa
python part2/diagonalization.py
```

### Chạy notebook

```bash
jupyter notebook
# Mở part1/part1_demo.ipynb hoặc part3/analysis.ipynb
```

### Cấu trúc kết quả chạy (ví dụ `decomposition.py`)

```
============================================================================
TEST SUITE: PHÂN RÃ GIÁ TRỊ SUY BIẾN (SVD)
============================================================================
 Ma trận vuông full-rank 3x3                   PASSED  (err = 2.22e-15)
 Ma trận chữ nhật ngang 2x3                    PASSED  (err = 8.88e-16)
 Ma trận chữ nhật dọc 3x2                      PASSED  (err = 8.88e-16)
 Ma trận suy biến (hai dòng phụ thuộc)         PASSED  (err = 8.88e-16)
 Ma trận toàn 0 kích thước 3x4                 PASSED  (err = 0.00e+00)
 Ma trận đơn vị 4x4                            PASSED  (err = 0.00e+00)
 Ma trận đường chéo có singular values lặp     PASSED  (err = 0.00e+00)
 Ma trận chứa số âm                            PASSED  (err = 4.44e-16)
 Ma trận kích thước 1x1                        PASSED  (err = 0.00e+00)
 Ma trận 1xN                                   PASSED  (err = 4.44e-16)
 Ma trận Nx1                                   PASSED  (err = 0.00e+00)
 Ma trận số rất nhỏ gần EPSILON                PASSED  (err = 2.00e-13)
 Ma trận Hilbert 3x3 (kiểm tra độ ổn định s... PASSED  (err = 6.69e-14)
 Dữ liệu rỗng                                  PASSED  (Bắt đúng lỗi: Ma trận A không được rỗng)
 Ma trận có 0 cột                              PASSED  (Bắt đúng lỗi: Ma trận A phải có ít nhất 1 cột)      
 Ma trận không cùng số cột                     PASSED  (Bắt đúng lỗi: Tất cả các dòng của A phải cùng số cột)
---------------------------------------------------------------------------
 TỔNG KẾT: 16/16 PASSED
```

---

## Báo cáo

Báo cáo kỹ thuật được viết bằng LaTeX, nằm trong thư mục `report/`.

**Biên dịch báo cáo:**

```bash
cd report
pdflatex report.tex
pdflatex report.tex   # Chạy lần 2 để cập nhật references
```

File báo cáo đã biên dịch: [`report/report.pdf`](report/report.pdf)

---

## Ghi chú kỹ thuật

### Hằng số EPSILON

```python
# config.py
EPSILON = 1e-12

def is_zero(x: float) -> bool:
    return abs(x) < EPSILON

def zero_rectify(value: float) -> float:
    return 0.0 if is_zero(value) else value
```

`EPSILON = 1e-12` được chọn là ngưỡng xử lý sai số số thực trong toàn bộ dự án. Giá trị nhỏ hơn machine epsilon của `float64` (~2.2e-16) trong một số tình huống, tuy nhiên đủ chặt để loại bỏ nhiễu số học trong hầu hết bài toán thực tế.

### Độ ổn định số học trong SVD

- **Numpy không được dùng** để tính SVD, trị riêng hay bất kỳ phép phân rã nào.
- Thuật toán Jacobi đảm bảo hội tụ cho ma trận đối xứng nhờ bất biến Frobenius: tổng bình phương các phần tử off-diagonal giảm đơn điệu sau mỗi vòng quay.
- Sau khi tính V qua Jacobi, `_orthogonalize()` được áp dụng bổ sung để đảm bảo trực giao chặt chẽ trong trường hợp trị riêng trùng (degenerate eigenspace).
- Với U: nếu $\sigma_i \approx 0$ thì $u_i$ không tính theo công thức $u_i = Av_i/\sigma_i$ mà được chọn từ cơ sở trực giao bổ sung để hoàn chỉnh ma trận $U$ thành trực giao $m \times m$.

### Hạn chế đã biết

- **Trị riêng phức:** `diagonalize()` ném `ValueError` nếu ma trận có trị riêng phức (không xử lý trường hợp phức).
- **Khối Jordan:** Ma trận không chéo hóa được trên $\mathbb{R}$ (thiếu vector riêng độc lập) sẽ ném `ValueError`.
- **Ma trận lớn:** Do dùng Python thuần, hiệu năng thấp hơn NumPy đáng kể với ma trận kích thước lớn (> 100×100).
- **Thuật toán QR trong chéo hóa:** Có thể không hội tụ sau 4000 vòng lặp với ma trận đặc biệt — ném `ValueError` trong trường hợp này.

---

## Tác giả & Đóng góp

| MSSV | Họ và Tên | GitHub |
|-----------|--------|--------|
| 24120018 | Đào Thị Quỳnh Anh | [@Meg-0911](https://github.com/Meg-0911) |
| 24120347 | Phan Lê Đăng Khoa | [@EndlessMelody](https://github.com/EndlessMelody) |
| 24120348 | Hà Đăng Khôi | [@hakhoi1901](https://github.com/hakhoi1901) |
| 24120352 | Lương Trung Kiên | [@kienluongisme](https://github.com/kienluongisme) |
| 24120418 | Nguyễn Minh Quân | [@SlimeQzz2606](https://github.com/SlimeQzz2606)|
---

*Dự án được thực hiện trong khuôn khổ môn học, năm học 2025–2026.*
