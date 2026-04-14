# Phần 3: Giải Hệ Phương Trình và Phân Tích Hiệu Năng

Thư mục này chứa mã nguồn thực hiện benchmark hiệu năng và độ ổn định của 3 phương pháp giải hệ phương trình tuyến tính $Ax = b$:
1. **Gauss Elimination** (`gauss`) - Sử dụng `numpy.linalg.solve` (Fallback sang `np.linalg.lstsq` khi ma trận suy biến).
2. **Singular Value Decomposition** (`svd`) - Sử dụng `numpy.linalg.lstsq` (Pseudo-inverse).
3. **Gauss-Seidel** (`gauss_seidel`) - Sử dụng **Pure Python** với vòng lặp lặp (tối đa 1000 vòng lặp).

---

##  Jupyter Notebook (`analysis.ipynb`)

Mục tiêu chính trong phần Notebook là **đọc dữ liệu đã xuất sẵn** và **vẽ đồ thị phân tích**, không cần phải code lại các thuật toán tính toán.

### 1. Dữ liệu Đầu Vào
Toàn bộ dữ liệu bạn cần đã được script chạy và lưu vào file **`benchmark_results.json`**. 
Cấu trúc của JSON file bao gồm 2 phần chính:

```json
{
  "performance": {
    "gauss": {
      "50": { "time_s": 0.0001, "error": 2.5e-16 },
      "100": { ... },
      "200": { ... },
      "500": { ... },
      "1000": { ... }
    },
    "svd": { ... },
    "gauss_seidel": { ... }
  },
  "stability": {
    "50": {
      "hilbert": {
        "cond": 1.099e+19,
        "gauss_err": 5.58e-16,
        "svd_err": 8.46e-16,
        "gs_err": "ERR"
      },
      "spd": {
        "cond": 2.426,
        "gauss_err": 2.47e-16,
        "svd_err": 7.83e-16,
        "gs_err": 1.20e-12
      }
    },
    "100": { ... },
    ...
  }
}
```

### 2. Các Biểu Đồ Cần Vẽ
Sử dụng `matplotlib` / `seaborn` và `pandas` để trực quan hoá:

1. **Đồ thị Log-Log Hiệu Năng:**
   - **X-axis (Log scale):** Kích thước ma trận $n$ (50, 100, 200, 500, 1000).
   - **Y-axis (Log scale):** Thời gian chạy `time_s` (Lấy từ `["performance"]`).
   - Yêu cầu vẽ 3 đường (Gauss, SVD, Gauss-Seidel) cùng với **2 đường tham chiếu lý thuyết (Reference lines): đường $y = n^3$ và đường $y = n^2$**.
   - *Lý luận phân tích:* Trong đồ thị Log-Log, hàm đa thức sẽ biến thành đường thẳng, độ dốc (slope) của đường thẳng chính là số mũ. Hãy dùng đồ thị này để chứng minh cho Giảng viên thấy:
     + Đường của **Gauss và SVD** có độ dốc song song với đường $y = n^3$ (Vì đây là phương pháp trực tiếp $O(n^3)$).
     + Đường của **Gauss-Seidel** có độ dốc thấp hơn, song song với đường $y = n^2$ (Vì đây là phương pháp lặp, chi phí cho một lần quét ma trận chỉ là $O(n^2)$).
   
2. **So sánh Độ Ổn Định Số Học (Ma trận Well-conditioned vs Ill-conditioned):**
   - Lấy dữ liệu từ block `["stability"]`.
   - **Bảng/Bar chart so sánh Error:** Đối chiếu sai số tương đối (`gauss_err`, `svd_err`) giữa hai loại ma trận: **Hilbert** (số điều kiện rất cao, Ill-conditioned) và **SPD** (số điều kiện thấp, Well-conditioned).
   - *Lưu ý:* `gs_err` của ma trận Hilbert mang giá trị chuỗi `"ERR"`, bởi vì ma trận Hilbert không chéo trội nên Gauss-Seidel bị từ chối giải (tránh lặp vô hạn). Hãy khéo léo dùng ghi chú "Diverge / Không hội tụ" trong bài thuyết trình.

### 3. Cách chạy lại file Benchmark (Nếu cần số liệu mới)

Nếu cần lấy lại tập số liệu thống kê trung bình khác, hãy mở Terminal tại thư mục gốc của project (nơi có file `config.py`) và gõ:

```bash
python -X utf8 part3/benchmark.py
```

---

## Các File Xử Lý Cốt Lõi

- **`solvers.py`**: Chứa dispatcher hàm `solve_system()` phân định phương pháp giải toán cho `method='gauss'`, `'svd'` hay `'gauss_seidel'`. Có check chéo trội (Diagonally dominant) ở đây.
- **`data_gen.py`**: Các thuật toán sinh dữ liệu Matrix test. Đặc biệt có thuật toán tự bắt ép ma trận bất kì phải thành ma trận SPD chéo trội nghiêm ngặt.
- **`benchmark.py`**: Script điều phối chạy thử nghiệm hiệu năng thực tế 5 vòng, đo đạc sai số và xuất file kết quả tổng hợp.
- **`time_benchmark.py`**: Bản benchmark nhanh dùng in trực quan thời gian ra output terminal. 
