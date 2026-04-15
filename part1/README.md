# Phần 1: Các phép toán Đại số tuyến tính cơ bản

Phần này tập trung vào việc cài đặt thủ công (from scratch) các thuật toán cơ bản của Đại số tuyến tính. Chúng tôi không sử dụng các thư viện như `numpy` hay `scipy` ở phần cài đặt thuật toán nhằm hiểu rõ hơn về mặt tính toán và quản lý lỗi số học cơ bản (với các hàm do nhóm tự xây dựng). Các thư viện ngoài chỉ được dùng để so sánh qua các bài kiểm thử (unit tests).

## 🗂 Các chức năng cốt lõi

- **`gaussian.py`**: Cài đặt phương pháp khử Gauss (Gaussian Elimination) với kỹ thuật chọn phần tử chốt cột (Partial Pivoting). Hàm `gaussian_eliminate` nhận vào ma trận hệ số $A$ và vector $b$, trả về hệ đã được đưa về dạng ma trận bậc thang. Đảm bảo tính ổn định số học và giảm thiểu sai số nhờ partial pivoting.
- **`back_substitution.py`**: Cài đặt thuật toán thế ngược (Back Substitution). Chuyên xử lý giải nghiệm cho ma trận tam giác trên lấy từ `gaussian.py`, ngoài ra cũng có khả năng phát hiện hệ vô số nghiệm và in ra công thức nghiệm tổng quát.
- **`determinant.py`**: Tính định thức của ma trận vuông bằng cách sử dụng lại thuật toán khử Gauss. Định thức được tính bằng tích các phần tử trên đường chéo chính sau khi biến đổi, có hiệu chỉnh dấu tùy thuộc vào số lần hoán vị dòng.
- **`inverse.py`**: Cài đặt phương pháp Gauss-Jordan mở rộng (đưa ma trận bổ sung $[A|I]$ về RREF) để tìm ma trận nghịch đảo $A^{-1}$.
- **`rank_basis.py`**: Thuật toán tính Hạng (Rank) của ma trận, đồng thời trích xuất các vector cơ sở cho Không gian cột (Column Space), Không gian hàng (Row Space) và Không không gian nghiệm (Null Space).
- **`verify_solution.py`**: Đánh giá sai số của nghiệm thu được (Residual Error) dựa trên chuẩn L2 ($||b - Ax||_2$).

## Hướng dẫn sử dụng

Bạn có thể chạy thử nghiệm qua Notebook minh hoạ đã được tạo sẵn:
- **`part1_demo.ipynb`**: Chứa các ví dụ cụ thể cho mỗi phép tính và hướng dẫn trực quan.

## Kiểm thử (Testing)

Mỗi mô-đun đều đi kèm với một hàm kiểm thử tương ứng để đảm bảo độ chính xác trên nhiều dạng ma trận (đặc biệt các ma trận suy biến, ma trận không vuông). 

### Chạy từng module riêng lẻ

Bạn có thể kiểm tra trực tiếp từng thuật toán bằng cách chạy độc lập file `.py` tương ứng. Từng file đều được tích hợp sẵn các bộ kiểm tra nhỏ (`test_cases`) và khối `if __name__ == "__main__":`. Ví dụ:
```bash
python gaussian.py             # Chỉ test thuật toán Khử Gauss
python back_substitution.py    # Chỉ test chạy Thế ngược
python determinant.py          # Chỉ test thuật toán Tìm định thức
python inverse.py              # Chỉ test thuật toán Tìm ma trận nghịch đảo
python rank_basis.py           # Chỉ test thuật toán Tìm Hạng và Cơ sở
python verify_solution.py      # Chỉ test Kiểm chứng nghiệm
```

### Chạy toàn bộ kiểm thử

Để chạy tất cả các bài kiểm thử cục bộ cùng lúc cho toàn bộ Phần 1, bạn sử dụng lệnh:
```bash
python run_all_tests.py
```
