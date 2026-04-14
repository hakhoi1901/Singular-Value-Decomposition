### CẤU TRÚC BÁO CÁO: PHẦN 3

#### 3. Phần 3: Giải Hệ Phương Trình và Phân Tích Hiệu Năng

**3.1. Thiết lập thực nghiệm (Experimental Setup)**
* **3.1.1. Cấu hình phần cứng và phần mềm:**
    * (Agent cần điền: OS, CPU, RAM, phiên bản Python, phiên bản thư viện NumPy).
    * *Mục đích:* Đảm bảo tính lặp lại (reproducibility) của thí nghiệm đo thời gian.
* **3.1.2. Môi trường kiểm thử và Phương pháp đo lường:**
    * *Mô tả cách đo thời gian:* Sử dụng `time.perf_counter()`. Trình bày phương pháp lặp 5 lần lấy trung bình (Average over 5 runs) để loại bỏ nhiễu từ hệ điều hành.
    * *Mô tả cách đo sai số:* Sai số tương đối (Relative Residual Error) được tính bằng công thức $e = \frac{||A\hat{x} - b||_2}{||b||_2}$, trong đó chuẩn $L_2$ được tính an toàn thông qua `np.linalg.norm` nhằm tránh hiện tượng tràn số (overflow) trên các ma trận kích thước lớn.
* **3.1.3. Cơ chế Fallback và Ứng dụng NumPy:**
(`np.linalg.solve`) và SVD (`np.linalg.lstsq`) để bảo toàn tính đồng nhất (Apples-to-Apples) khi vẽ đồ thị $O(n^3)$. Riêng Gauss-Seidel vẫn giữ nguyên bản cài đặt Python thuần của nhóm do tính hội tụ nhanh của nó. Bổ sung việc bọc lỗi (try-except) `LinAlgError: Singular matrix` của Gauss để tự động chuyển sang giải bằng bình phương tối thiểu (lstsq).

**3.2. Phân tích Hiệu năng (Performance Analysis)**
* **3.2.1. Cơ sở lý thuyết về Độ phức tạp thuật toán:**
    * *Khử Gauss & Phân rã SVD:* Nhắc lại lý thuyết cả hai đều thuộc lớp thuật toán trực tiếp (Direct Methods) với chi phí tính toán $\mathcal{O}(n^3)$. Tuy nhiên, hằng số (constant factor) của SVD lớn hơn do đòi hỏi nhiều phép quay và tìm trị riêng.
    * *Gauss-Seidel:* Nhắc lại đây là phương pháp lặp (Iterative Method), độ phức tạp cho mỗi vòng lặp là $\mathcal{O}(n^2)$.
* **3.2.2. Kết quả đo lường thời gian thực thi:**
    * *Bảng 3.1: Thời gian trung bình thực thi (giây) theo kích thước $n$ trên ma trận SPD.*
    * *(Để trống: Bạn chèn Bảng 1 từ Output Terminal vào đây)*
* **3.2.3. Đồ thị Log-Log và Đối chiếu Thực nghiệm:**
    * *(Để trống: Bạn chèn Hình ảnh Biểu đồ Log-Log vào đây. Yêu cầu biểu đồ có 3 đường thực tế và 2 đường tham chiếu $y=n^3$, $y=n^2$)*
    * *Nhận xét:*
        * Phân tích sự song song giữa đường của Gauss/SVD với đường tham chiếu $O(n^3)$.
        * Phân tích sự chênh lệch (gap) giữa hai đường Gauss và SVD để chứng minh SVD có hằng số tính toán cao hơn.
        * Đặc biệt nhấn mạnh việc đường Gauss-Seidel song song với đường $O(n^2)$, chứng thực được tốc độ vượt trội của phương pháp lặp trên hệ sparse/chéo trội so với phương pháp trực tiếp.

**3.3. Phân tích Độ Ổn định Số học (Numerical Stability Analysis)**
* **3.3.1. Cơ sở lý thuyết:**
    * *Định nghĩa Số điều kiện (Condition Number):* Trình bày công thức $\kappa_p(A) = ||A||_p \cdot ||A^{-1}||_p$. Nêu rõ ý nghĩa: $\kappa(A)$ là hệ số khuếch đại sai số.
    * *Định lý Sai số (Error bound):* Trích dẫn công thức $\frac{||\delta x||}{||x||} \le \kappa(A) \frac{||\delta b||}{||b||}$ [1]. Giải thích cơ chế "Ill-conditioned" (Điều kiện kém): $\kappa(A)$ càng lớn, một nhiễu nhỏ trong dữ liệu đầu vào ($A$ hoặc $b$) sẽ gây ra sai số khổng lồ ở nghiệm đầu ra ($x$).
* **3.3.2. Thiết lập hai đối tượng khảo sát:**
    * *Ma trận SPD (Well-conditioned):* Cách nhóm sinh ma trận SPD (nhấn mạnh việc ép ma trận thành dạng chéo trội nghiêm ngặt - Strictly Diagonally Dominant).
    * *Ma trận Hilbert $H_n$ (Ill-conditioned):* Định nghĩa ma trận Hilbert. Đây là bài test khắc nghiệt nhất cho độ ổn định số học.
* **3.3.3. Kết quả đo lường Sai số tương đối:**
    * *Bảng 3.2: So sánh số điều kiện $\kappa_2(A)$ và Sai số tương đối giữa ma trận Hilbert và ma trận SPD.*
    * *(Để trống: Bạn chèn Bảng 2 từ Output Terminal vào đây)*
* **3.3.4. Nhận xét và Lý giải Hiện tượng:**
    * *Phân tích SPD:* Nhận xét sai số của Gauss và SVD luôn giữ ở mức nhiễu máy (Machine Epsilon $\approx 10^{-15}$). Gauss-Seidel hội tụ ổn định.
    * *Hiện tượng Bùng nổ Sai số (Error Explosion) trên Hilbert:* Chỉ ra con số sai số lên tới $10^5, 10^8$ của thuật toán Gauss. Áp dụng trực tiếp Định lý sai số (Error bound) để giải thích sự bùng nổ này đến từ số điều kiện $\kappa(A) \approx 10^{20}$ của ma trận Hilbert.
    * *Sự ưu việt của SVD:* Nhận xét tại sao SVD vẫn duy trì được sai số nhỏ ($10^{-14}$) trên ma trận Hilbert. Lý do: Tính chất nghịch đảo giả (Pseudo-inverse) và cơ chế cắt tỉa (truncate) các giá trị singular tiệm cận 0 giúp loại bỏ hoàn toàn các thành phần gây nhiễu, làm mượt (regularization) cấu trúc ma trận.
    * *Hạn chế của Gauss-Seidel:* Phân tích nguyên nhân Gauss-Seidel báo `ERR` trên ma trận Hilbert (Do ma trận không thỏa mãn điều kiện đủ là chéo trội chặt hàng, dẫn đến bán kính phổ lớn hơn 1 $\rightarrow$ phân kỳ).

**3.4. Tổng kết và Đánh giá Lựa chọn Phương pháp**
* (Agent tổng hợp lại ưu nhược điểm)
    * **Khử Gauss (LU):** Nhanh nhất, rẻ nhất ($O(n^3)$ chuẩn), nhưng yếu đuối nhất trước ma trận điều kiện kém. Lựa chọn hàng đầu cho các hệ thông thường.
    * **Phân rã SVD:** Đắt đỏ nhất về thời gian thực thi, nhưng sở hữu khả năng "miễn nhiễm" đáng kinh ngạc với sai số số học. Lựa chọn bắt buộc cho các hệ Ill-conditioned hoặc bài toán bình phương tối thiểu.
    * **Lặp Gauss-Seidel:** Tốc độ thần tốc tiệm cận $O(n^2)$ trên ma trận chéo trội, tiết kiệm bộ nhớ. Nhưng điều kiện áp dụng quá khắt khe, dễ dàng gãy đổ nếu ma trận không đáp ứng tiêu chuẩn.

---

**Tài liệu Tham khảo (References cho phần 3)**
[1] Lloyd N. Trefethen and David Bau III, *Numerical Linear Algebra*, SIAM, 1997. (Sách kinh điển về giải tích số, nơi chứa định lý sai số và số điều kiện).
[2] Gene H. Golub and Charles F. Van Loan, *Matrix Computations*, 4th Edition, Johns Hopkins University Press, 2013. (Nguồn giải thích tính ổn định của SVD).

---
