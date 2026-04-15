# Phần 3: Giải Hệ Phương Trình và Phân Tích Hiệu Năng

Thư mục này chứa mã nguồn benchmark 3 phương pháp giải hệ $Ax = b$ và **hướng dẫn dựng file `analysis.ipynb`**.

---

## Các file có sẵn

| File | Vai trò |
|---|---|
| `solvers.py` | Dispatcher `solve_system(A, b, method)` — gọi đúng backend (NumPy hoặc Pure Python) |
| `data_gen.py` | Sinh ma trận SPD (well-conditioned) và Hilbert (ill-conditioned) |
| `benchmark.py` | Đo hiệu năng + độ ổn định, lặp 5 lần lấy trung bình, xuất `benchmark_results.json` |
| `plot_chart.py` | Đọc JSON và vẽ đồ thị Log-Log, lưu `benchmark_plot.png` |
| `time_benchmark.py` | In bảng thời gian nhanh ra terminal |
| `benchmark_results.json` | **Dữ liệu đã chạy sẵn** — không cần chạy lại nếu không muốn |
| `analysis.ipynb` | ← **File bạn cần tạo** (xem hướng dẫn bên dưới) |

Chạy lại benchmark nếu cần (từ thư mục gốc project):

```bash
python -X utf8 part3/benchmark.py   # sinh dữ liệu mới
python part3/plot_chart.py           # vẽ lại đồ thị
```

---

## Hướng dẫn dựng `analysis.ipynb`

> **Đầu ra mong đợi:** File notebook hoàn chỉnh, chạy từ đầu tới cuối không lỗi, có thể trình bày thẳng cho Giảng viên.

Notebook được tổ chức theo **5 phần** (dùng Markdown cell để phân cách phần):

---

### Phần 1 — Giới thiệu và đọc dữ liệu

**[Markdown cell]** Tiêu đề phần + giới thiệu ngắn (1-2 dòng về mục tiêu của notebook).

**[Code cell]** Import thư viện và đọc `benchmark_results.json`:

```python
# import thư viện
# đọc benchmark_results.json
# in ra kích thước ma trận
# in ra thời gian thực thi của 3 phương pháp
# in ra số điều kiện của 2 loại ma trận
# in ra sai số của 3 phương pháp
```

---

### Phần 2 — Cơ sở lý thuyết (Markdown cell, không cần code)

Trình bày **ngắn gọn** hai công thức toán học cốt lõi (không cần chứng minh, chỉ cần nêu và giải thích):

**2.1. Số điều kiện (Condition Number)**

$$\kappa_p(A) = \|A\|_p \cdot \|A^{-1}\|_p = \frac{\sigma_{\max}(A)}{\sigma_{\min}(A)}$$

Ý nghĩa: $\kappa_2(A)$ là hệ số khuếch đại sai số. Ma trận Hilbert có $\kappa_2 \sim 10^{20}$ — nghĩa là sai số nhỏ trong dữ liệu đầu vào có thể bị khuếch đại $10^{20}$ lần ở nghiệm.

**2.2. Công thức lặp Gauss-Seidel**

$$x_i^{(k+1)} = \frac{1}{a_{ii}} \left( b_i - \sum_{j < i} a_{ij} x_j^{(k+1)} - \sum_{j > i} a_{ij} x_j^{(k)} \right)$$

Điều kiện hội tụ: ma trận $A$ phải **chéo trội nghiêm ngặt** theo hàng: $|a_{ii}| > \sum_{j \neq i} |a_{ij}|$.

---

### Phần 3 — Đồ thị Log-Log (Phân tích hiệu năng)

**[Code cell]** Trích xuất thời gian và vẽ đồ thị:

```python
# trích xuất thời gian của 3 phương pháp
# vẽ đồ thị log-log (gọi hàm plot_chart)
```

**[Markdown cell]** Nhận xét đối chiếu lý thuyết:

- **Gauss và SVD** có độ dốc song song với đường $O(n^3)$ → xác nhận cả hai đều là phương pháp trực tiếp bậc 3. Khoảng cách dọc giữa hai đường phản ánh hằng số tính toán của SVD lớn hơn Gauss (do SVD cần nhiều phép quay hơn LU).
- **Gauss-Seidel** có độ dốc **thấp hơn rõ rệt**, song song với đường $O(n^2)$ → xác nhận chi phí mỗi vòng lặp chỉ là $O(n^2)$.
- ⚠️ **Lưu ý quan trọng:** Đường Gauss-Seidel (xanh lá) đang nằm *phía trên* đường Gauss (xanh dương) vì được cài đặt bằng Python thuần — hằng số tuyệt đối cao hơn nhiều so với backend C/Fortran của NumPy. Tuy nhiên, **độ dốc của nó thấp hơn hẳn** (bậc 2 thay vì bậc 3). Nếu tiếp tục tăng $n$ lên $10{,}000$ hoặc $100{,}000$, đường xanh lá chắc chắn sẽ **cắt và đi xuống dưới** đường xanh dương — chứng minh rằng phương pháp lặp vượt trội hoàn toàn so với phương pháp trực tiếp ở quy mô siêu lớn nếu được cài đặt tối ưu.

---

### Phần 4 — Bảng phân tích độ ổn định số học

**[Code cell]** Dựng DataFrame và hiển thị bảng:

```python
# trích xuất sai số của 3 phương pháp
# in ra số điều kiện của 2 loại ma trận
# in ra bảng
```

**[Markdown cell]** Giải thích Định lý 3.1:

**Định lý sai số (Error Bound):** Cho $Ax = b$ là nghiệm đúng và $A\hat{x} = b + \delta b$ là nghiệm bị nhiễu. Khi đó:

$$\frac{\|\hat{x} - x\|}{\|x\|} \leq \kappa_2(A) \cdot \frac{\|\delta b\|}{\|b\|}$$

Áp dụng vào kết quả bảng:

- **Ma trận SPD:** $\kappa_2 \approx 2\text{–}3$ → sai số nghiệm bị khuếch đại tối đa $3\times$ → hoàn toàn chấp nhận được, giải thích vì sao cả Gauss lẫn SVD đều cho residual $\sim 10^{-16}$.
- **Ma trận Hilbert:** $\kappa_2 \approx 10^{20}$ → sai số nghiệm có thể bị khuếch đại $10^{20}$ lần → với nhiễu làm tròn $\varepsilon_{\text{machine}} \approx 10^{-16}$, sai số nghiệm lên tới $10^4$. Residual nhỏ ($\sim 10^{-15}$) chỉ là **ảo ảnh** — không phản ánh độ chính xác của nghiệm.
- **Gauss-Seidel / Hilbert = ERR:** Ma trận Hilbert không thỏa mãn điều kiện chéo trội chặt → `is_strictly_diagonally_dominant` trả về `False` → `ValueError` được ném ngay, không thực hiện bất kỳ vòng lặp nào.

---

### Phần 5 — Nhận xét và kết luận

**[Markdown cell]** Tổng hợp mối quan hệ giữa **chi phí tính toán** và **tính ổn định số**:

| Phương pháp | Chi phí | Ổn định (SPD) | Ổn định (Hilbert) | Điều kiện áp dụng |
|---|---|---|---|---|
| Khử Gauss (LU) | $O(n^3)$ thấp nhất | $\sim\varepsilon_{\text{machine}}$ | Residual giả | Mọi ma trận không suy biến |
| Phân rã SVD | $O(n^3)$ hằng số lớn | $\sim\varepsilon_{\text{machine}}$ | Tốt nhất | Mọi ma trận (kể cả singular) |
| Gauss-Seidel | $O(kn^2)$, $k \ll n$ | $\sim 10^{-12}$ | ERR | Chỉ ma trận chéo trội chặt |

**Kết luận:**
1. **Không có phương pháp nào là tốt nhất tuyệt đối** — mỗi phương pháp có miền ứng dụng tối ưu riêng.
2. **Chi phí $O(n^3)$ không nói lên tất cả** — hằng số ẩn của SVD lớn hơn Gauss $\approx 7\times$ ở $n=1000$, nhưng SVD là công cụ duy nhất đáng tin cậy trên hệ ill-conditioned.
3. **Gauss-Seidel chứng minh sức mạnh tiệm cận của $O(n^2)$** — mặc dù chậm hơn ở kích thước hiện tại do Python thuần, độ dốc trên đồ thị Log-Log xác nhận rằng ở quy mô $n \geq 10{,}000$, phương pháp lặp sẽ vượt các phương pháp trực tiếp nếu được cài đặt tối ưu.
