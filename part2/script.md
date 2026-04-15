# KỊCH BẢN THUYẾT TRÌNH: PHÂN TÍCH Singular Value Decomposition (SVD)

## Khúc dạo đầu
- **Visuals:** Hiện tiêu đề chính: "Singular Value Decomposition (SVD)". Sau đó xuất hiện dòng phụ đề "Chapter 1 — Anatomy of Chaos" và công thức cốt lõi nhấp nháy $A = U \Sigma V^T$.
- **Chuyển cảnh:** Tiêu đề mờ dần (Fade out), camera đưa người xem vào không gian 2D với lưới tọa độ (NumberPlane).

---

## Cảnh 1: Giải phẫu sự hỗn loạn (Anatomy of Chaos)
**Mục tiêu:** Cho thấy khi một ma trận biến đổi không gian, nó sẽ làm mất đi tính trực giao (vuông góc) của các vector cơ sở, từ đó dẫn đến nhu cầu tìm kiếm SVD.

| Thời điểm | Hiệu ứng hình ảnh (Visuals) | Phụ đề / Lời bình (Voice-over) |
| :--- | :--- | :--- |
| **Beat 1.1** | Camera vẽ lên không gian hệ tọa độ XY trắng xám. <br>Xuất hiện vòng tròn đơn vị ($\|x\|_2 = 1$). <br>Vẽ 2 vector cơ sở màu đỏ ($\hat{i}$) và xanh lục ($\hat{j}$) kèm ký hiệu góc vuông 90 độ. | "Để giải mã một phép biến đổi phức tạp, chúng ta phải bắt đầu từ những thứ cơ bản nhất." <br><br>"Hãy quan sát không gian hai chiều này. Bắt đầu từ đường tròn đơn vị, nơi mọi vector đều có chiều dài chuẩn xác bằng 1." <br><br>"Tại lõi của nó, hai vector cơ sở $i$ và $j$ tạo thành một góc vuông. Đây là nền tảng của không gian." |
| **Beat 1.2** | Ở góc trái màn hình, xuất hiện ma trận $A = \begin{bmatrix} 3 & 1 \\ 0 & 2 \end{bmatrix}$. <br>Hệ tọa độ bị kéo giãn. Vòng tròn biến thành một **Hình Elip màu vàng** (Image of unit circle under A). Các vector $i, j$ bị nghiêng đi. | "Xét ma trận A = [[3,1],[0,2]]. Đây là một phép biến đổi tuyến tính đơn giản." <br><br>"A kéo giãn và xoay không gian — đường tròn đơn vị bị biến dạng thành một hình elip. Hình elip này chứa đựng toàn bộ nội dung hình học của phép biến đổi A." |
| **Beat 1.3** | Tập trung vào 2 vector ảnh. Hiển thị thông số góc lệch $\theta$ giữa hai vector. <br>Viết ra phép tính tích vô hướng: $\langle A\hat{i},\, A\hat{j} \rangle = 3 \neq 0$ và đóng khung đỏ số $0$. | "Nhưng có điều gì đó đã bị phá vỡ. Góc vuông ban đầu giữa $\hat{i}$ và $\hat{j}$ không còn nữa. <br><br>Tích vô hướng bằng 3 (khác 0). Hai vector ảnh không còn trực giao với nhau." |
| **Beat 1.4** | Bỏ đi các vector bị méo, vẽ hai trục chính của hình elip (Dashed lines: $\sigma_1 u_1$ và $\sigma_2 u_2$). Camera làm nổi bật việc 2 trục này **vuông góc hoàn hảo**. <br>Hiện dòng chữ: "How to recover orthogonality?" và công thức SVD. | "Tuy nhiên, elip luôn có các trục chính — và các trục này lại vuông góc với nhau! <br><br>Câu hỏi đặt ra: Liệu có tồn tại một hệ cơ sở đầu vào mà $A$ biến đổi thành một hệ trực giao?" <br><br>"SVD chính là câu trả lời: $A = U\Sigma V^T$ — phân tách phép biến đổi thành xoay, giãn, rồi xoay lại." |

---

## Cảnh 2: Trực quan hóa hình học (Geometric Interpretation)
**Mục tiêu:** Cụ thể hoá định nghĩa "xoay, giãn, xoay lại" bằng hoạt ảnh phân rã từng bước một từ phải sang trái.

| Thời điểm | Hiệu ứng hình ảnh (Visuals) | Phụ đề / Lời bình (Voice-over) |
| :--- | :--- | :--- |
| **Giới thiệu** | Đưa 3 ma trận $U, \Sigma, V^T$ lên màn hình. Focus vào từng ma trận theo thứ tự từ phải về bên trái. | "Vậy chính xác thì ba ma trận $U, \Sigma, V^T$ làm gì với không gian của chúng ta? Hãy phân tích lại cỗ máy này, bắt đầu từ phải sang trái." |
| **Bước 1: $V^T$** | Không gian và lưới tọa độ bị xoay đi một góc. Hình tròn hiển thị ở giữa vẫn giữ nguyên hình dạng tròn trịa hoàn hảo. | "Bước một, thao tác trên không gian nguồn với ma trận $V^T$. <br>Thử thay đổi góc xoay tham số của nó. Chú ý: Hình tròn vẫn là hình tròn. Nhiệm vụ duy nhất của nó là dò tìm và định vị hệ cơ sở chuẩn bị cho phép biến đổi." |
| **Bước 2: $\Sigma$** | Hệ không gian thay đổi phương dọc và ngang, hình tròn chính thức kéo dãn thành Elip. Đoạn biểu diễn $\Sigma_2 = 0$ làm Elip xẹp hoàn toàn thành đường thẳng. | "Bước hai: Ma trận đường chéo $\Sigma$ - sự co giãn. <br>Không gian thay đổi dọc theo trục hoành... và thay đổi dọc theo trục tung. <br><br>Nhưng điều gì xảy ra nếu ta ép $\Sigma_2$ về 0? Không gian bị biến đổi thành một đường thẳng. Chiều thứ hai đã mất." <br><br>*(Elip phình ra lại)*: "Giờ ta trả không gian về lại giá trị $\Sigma$ thật của bài toán." |
| **Bước 3: $U$** | Hình Elip bị xoay thêm một lần nữa đến vị trí khớp hoàn toàn với Elip của ma trận $A$ lúc đầu. | "Cuối cùng là $U$ - một ma trận trực giao khác tiếp quản. Tương tự, $U$ xoay khối elip này thực hiện phép quay." |
| **Tổng kết** | Chơi nhanh lại 3 animations: Xoay $\rightarrow$ Kéo giãn $\rightarrow$ Xoay. | "Xoay. Kéo giãn. Và xoay. Phép màu của SVD đã giải mã phép biến đổi $A$ hình dung thành 3 bước đơn giản." |

---

## Cảnh 3: Thuật toán và Toán học phía sau SVD
**Mục tiêu:** Cung cấp chi tiết cách tính các giá trị bằng đại số tuyến tính kết hợp minh họa trục truyến.

| Thời điểm | Hiệu ứng hình ảnh (Visuals) | Phụ đề / Lời bình (Voice-over) |
| :--- | :--- | :--- |
| **Hệ thống hóa** | Màn hình hiển thị quá trình từng bước tính toán. | "Cho ma trận A 2x2. Mục tiêu: phân rã $A = U \Sigma V^T$. SVD hoạt động được với MỌI ma trận — không cần vuông hay đủ vector riêng." |
| **Tìm $V^T$** | Giải phương trình ma trận $A^T A$ và det. | "Bước 1a — Tính $A^T A$ (đây là ma trận đối xứng, nửa xác định dương). <br>Bước 1b — Giải phương trình đặc trưng $det(A^T A - \lambda I) = 0$. <br>Bước 1c — Tìm eigenvectors của $A^T A$; tìm v1 tương ứng với $\lambda_1$, v2 tương ứng $\lambda_2$. <br>Ghép v1, v2 theo cột $\rightarrow$ ma trận $V$ (và lấy chuyển vị $\rightarrow V^T$)." |
| **Tìm $\Sigma$** | Rút căn bậc hai eigenvalues. | "Bước 1d — Singular values: $\sigma_i = \sqrt{\lambda_i}$. Xếp giảm dần vào đường chéo $\Sigma$." |
| **Tìm $U$** | Áp dụng công thức $u_i$. Bảng thông số U, $\Sigma$, $V^T$ hiện ra hoàn chỉnh. | "Bước 1e — Tính left-singular vectors: $u_1 = \frac{A v_1}{\sigma_1}$ và $u_2 = \frac{A v_2}{\sigma_2}$. <br>Ghép $u_1, u_2$ theo cột $\rightarrow$ ma trận trực giao $U$. <br>Kết hợp lại: $A = U \Sigma V^T$ — phân rã SVD hoàn chỉnh!" |
| **Mảnh ghép** | Animation chiếu lại ma trận ráp dính vào Ellipse. Các trục Ellipse khớp $\sigma_i \cdot u_i$. | "Tổng kết hình học: Nhìn toàn cảnh biến đổi. Các bán trục của ellipse = $\sigma_i \cdot u_i$. <br>SVD luôn tồn tại và duy nhất (với $\sigma_i > 0$) cho MỌI ma trận thực!" |

---

## Cảnh 4: SVD vs Diagonalization (Chéo hóa)
**Mục tiêu:** So sánh sự ưu việt của công cụ SVD so với phương pháp phân tích Giá trị riêng (Eigen Decomposition).

| Thời điểm | Hiệu ứng hình ảnh (Visuals) | Phụ đề / Lời bình (Voice-over) |
| :--- | :--- | :--- |
| **Eigenvalues** | Đưa các vector riêng (Eigenvectors) trải lên không gian. Làm nổi bật các vector riêng này **không hề vuông góc** với nhau (khác với hệ cơ sở của SVD luôn có $\perp$). | "Chuyển sang phương pháp Chéo hóa. Mục tiêu là tìm kiếm 'sự bất biến'. Bước 1: Tìm các Giá trị riêng bằng phương trình đặc trưng. Với mỗi lambda, ta tìm một không gian vector riêng tương ứng. <br><br>Các thành phần đại số đã sẵn sàng. Hãy đưa chúng lên không gian. Lưu ý: Hai vector này rõ ràng không vuông góc với nhau." |
| **Sự bất biến** | Biến đổi ma trận A. Lưới tọa độ biến dạng, nhưng hai Eigenvector chỉ co giãn độ dài, không hề bị chệch khỏi hướng ban đầu. | "Sức mạnh của chúng nằm ở đâu? Hãy quan sát khi ma trận A làm thay đổi không gian. Mọi thứ bị thay đổi, nhưng hai vector này vẫn nằm trên quỹ đạo ban đầu. Chúng chỉ bị kéo giãn với tỷ lệ chính là các Giá trị riêng (Eigenvalues)." |
| **Chéo hóa** | Lắp ráp các phương trình $A = P D P^{-1}$. | "Giờ ta lắp ráp chúng thành phương trình ma trận hoàn chỉnh. Ma trận P được ghép từ các Vector riêng... Ma trận đường chéo D chứa các Giá trị riêng... và cuối cùng là nghịch đảo của P. Phép phân tích Chéo hóa đã hoàn tất." |
| **Kết luận** | Cầm cân nảy mực: $A = PDP^{-1}$ mờ đi/Bị vỡ nhẹ (nếu gặp ma trận hình chữ nhật), trong khi khối $A = U\Sigma V^T$ vững trãi sáng lên. | "Tuy nhiên, Chéo hóa sụp đổ nếu ma trận không vuông hoặc thiếu vector độc lập. <br><br>Ngược lại, SVD chấp nhận mọi ma trận và cung cấp một hệ cơ sở trực giao. Đó là lý do SVD trở thành nền tảng tối thượng của Tính toán khoa học." |
