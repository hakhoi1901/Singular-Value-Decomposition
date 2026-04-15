"""
plot_chart.py - Đọc dữ liệu từ file JSON và vẽ đồ thị Log-log.
Không thực hiện tính toán lại, chỉ tập trung trực quan hóa dữ liệu.
"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

def plot_from_json():
    # 1. Đọc dữ liệu từ file JSON
    json_path = Path(__file__).resolve().parent / "benchmark_results.json"
    
    if not json_path.exists():
        print("Lỗi: Không tìm thấy file benchmark_results.json!")
        print("Hãy chạy file benchmark.py trước để sinh dữ liệu.")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 2. Trích xuất mảng dữ liệu
    perf_data = data["performance"]
    sizes = [50, 100, 200, 500, 1000]
    
    times_gauss = [perf_data["gauss"][str(n)]["time_s"] for n in sizes]
    times_svd = [perf_data["svd"][str(n)]["time_s"] for n in sizes]
    times_gs = [perf_data["gauss_seidel"][str(n)]["time_s"] for n in sizes]

    # 3. Cấu hình vẽ đồ thị log-log
    plt.figure(figsize=(10, 6))

    plt.loglog(sizes, times_gauss, marker="o", color="blue", linewidth=2, label="Khử Gauss (LU)")
    plt.loglog(sizes, times_svd, marker="s", color="orange", linewidth=2, label="Phân rã SVD")
    plt.loglog(sizes, times_gs, marker="^", color="green", linewidth=2, label="Gauss-Seidel")

    # 4. Vẽ đường tham chiếu lý thuyết O(n^3) và O(n^2)
    # Lấy điểm neo là Gauss tại n=100
    if times_gauss[1] > 0:
        c3 = times_gauss[1] / (sizes[1] ** 3)
        theo_n3 = [c3 * (x ** 3) for x in sizes]
        plt.loglog(sizes, theo_n3, linestyle="--", color="black", alpha=0.7, label=r"Lý thuyết $O(n^3)$")

    # Lấy điểm neo là Gauss-Seidel tại n=100
    if times_gs[1] > 0:
        c2 = times_gs[1] / (sizes[1] ** 2)
        theo_n2 = [c2 * (x ** 2) for x in sizes]
        plt.loglog(sizes, theo_n2, linestyle=":", color="red", alpha=0.7, label=r"Lý thuyết $O(n^2)$")

    # 5. Trang trí đồ thị
    plt.xlabel("Kích thước ma trận n (Log scale)")
    plt.ylabel("Thời gian thực thi (giây) (Log scale)")
    plt.title("Biểu đồ Log-Log: Hiệu năng các thuật toán giải Hệ phương trình")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    # 6. Lưu file
    output_path = os.path.join(os.path.dirname(__file__), "benchmark_plot.png")
    plt.savefig(output_path, dpi=300)
    print(f"✓ Đã vẽ và lưu biểu đồ thành công tại: {output_path}")

if __name__ == "__main__":
    plot_from_json()