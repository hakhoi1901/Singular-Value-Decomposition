from manim import *
import numpy as np


A_mat = np.array([[3.0, 1.0],
                  [0.0, 2.0]])

# Tính SVD thực sự
U_np, S_np, Vt_np = np.linalg.svd(A_mat)
V_np = Vt_np.T

# Kết quả (làm tròn đẹp)
# U  ≈ [[-0.9806, -0.1961], [-0.1961,  0.9806]]  (nhưng dùng giá trị thực)
# S  ≈ [3.1623, 1.8974]
# Vt ≈ [[-0.9806, -0.1961], [ 0.1961, -0.9806]]

def np_to_3x3(mat2x2):
    """Nâng ma trận 2x2 lên 3x3 để dùng với ApplyMatrix Manim."""
    m = np.eye(3)
    m[:2, :2] = mat2x2
    return m

# Ma trận biến đổi 3x3
A_3x3  = np_to_3x3(A_mat)
U_3x3  = np_to_3x3(U_np)
Vt_3x3 = np_to_3x3(Vt_np)
Sig_3x3 = np_to_3x3(np.diag(S_np))


def create_text_with_bg(text, font_size=20, color=WHITE, bg_opacity=0.8, font="Cambria", weight=NORMAL):
    txt_obj = MarkupText(text, font=font, font_size=font_size, color=color, weight=weight).set_z_index(1)
    bg = BackgroundRectangle(txt_obj, color=BLACK, fill_opacity=bg_opacity, buff=0.15).set_z_index(0)
    return VGroup(bg, txt_obj)


def latex_matrix(arr, fmt=".3f", font_size=38, color=WHITE):
    """Trả về MathTex của ma trận numpy 2x2."""
    rows = arr.shape[0]
    cols = arr.shape[1]
    entries = " \\\\ ".join(
        " & ".join(f"{arr[r, c]:{fmt}}" for c in range(cols))
        for r in range(rows)
    )
    return MathTex(rf"\begin{{bmatrix}} {entries} \end{{bmatrix}}",
                   font_size=font_size, color=color)

# ══════════════════════════════════════════════════════════════════
class Scene3_SVD(MovingCameraScene):
    # ─── helper: subtitle ───────────────────────────────────────
    def show_subtitle(self, text, run_time=1.0):
        subtitle = create_text_with_bg(text, font_size=18, color=WHITE, font="Cambria")
        subtitle.to_edge(DOWN, buff=0.4) 
        if hasattr(self, "current_subtitle"):
            self.play(Transform(self.current_subtitle, subtitle), run_time=run_time)
        else:
            self.play(FadeIn(subtitle, shift=UP*0.2), run_time=run_time)
            self.current_subtitle = subtitle

    def clear_subtitle(self, run_time=0.5):
        if hasattr(self, "current_subtitle"):
            self.play(FadeOut(self.current_subtitle, shift=DOWN*0.2), run_time=run_time)
            del self.current_subtitle

    # ─── helper: NumberPlane tươi ────────────────────────────────
    def fresh_plane(self):
        return NumberPlane(
            x_range=[-10, 10, 1], y_range=[-6, 6, 1],
            background_line_style={"stroke_color": GREY,
                                   "stroke_width": 1,
                                   "stroke_opacity": 0.35},
            axis_config={"stroke_color": GREY_B, "stroke_width": 1.5},
        )

    def z6_tex(self, *tex, font_size=34, color=WHITE):
        return MathTex(*tex, font_size=font_size, color=color).set_z_index(6)
        
    # ─── construct ───────────────────────────────────────────────
    def construct(self):
        self.beat_0_intro()
        self.beat_1_algebraic_svd()
        self.beat_2_geometric_v_rotation()
        self.beat_3_geometric_sigma_scale()
        self.beat_4_geometric_u_rotation()
        self.beat_5_unit_circle_full()
        self.beat_6_summary()

    # ══════════════════════════════════════════════════════════════
    #  BEAT 0 — Giới thiệu bài toán
    # ══════════════════════════════════════════════════════════════
    def beat_0_intro(self):
        # Tiêu đề chính
        title = MathTex(r"\text{Singular Value Decomposition:}\quad A = U\,\Sigma\,V^\top",
                        font_size=36, color=WHITE)
        title_bg = BackgroundRectangle(title, color=BLACK, fill_opacity=0.85, buff=0.15)
        self.title_grp = VGroup(title_bg, title).to_edge(UP, buff=0.3)
        self.play(FadeIn(self.title_grp, shift=DOWN * 0.2), run_time=1.0)

        # Hiển thị ma trận A
        lbl_A = MathTex(r"A =", font_size=44, color=ORANGE)
        mat_A = MathTex(r"\begin{bmatrix} 3 & 1 \\ 0 & 2 \end{bmatrix}",
                        font_size=44, color=ORANGE)
        grp_A = VGroup(lbl_A, mat_A).arrange(RIGHT, buff=0.2).move_to(ORIGIN)

        self.show_subtitle("Cho ma trận A 2×2. Mục tiêu: phân rã A = U Σ Vᵀ.", run_time=1.5)
        self.play(FadeIn(grp_A, shift=UP * 0.3), run_time=1.2)
        self.wait(1.5)

        # Ghi chú ba thành phần
        note_u = create_text_with_bg("U: ma trận trực giao (xoay)", font_size=16, color=BLUE_B)
        note_s = create_text_with_bg("Σ: ma trận đường chéo (co giãn)", font_size=16, color=YELLOW)
        note_v = create_text_with_bg("Vᵀ: ma trận trực giao (xoay)", font_size=16, color=GREEN_B)

        # Gom chúng lại và dàn hàng ngang ở phía dưới
        note = VGroup(note_u, note_s, note_v).arrange(RIGHT, buff=0.4)
        note.to_edge(DOWN, buff=1.2) # Đẩy xuống gần cạnh dưới màn hình

        self.play(FadeIn(note, shift=UP*0.2))
        self.wait(2)

        self.show_subtitle("SVD hoạt động được với MỌI ma trận — không cần vuông hay đủ vector riêng.", run_time=2.0)
        self.wait(1.5)

        self.play(FadeOut(grp_A), FadeOut(note), run_time=1.0)

     # ══════════════════════════════════════════════════════════════
    #  BEAT 1 — Tính toán đại số HOÀN CHỈNH
    #  1a  AᵀA
    #  1b  Phương trình đặc trưng → eigenvalues λ₁, λ₂
    #  1c  Eigenvectors của AᵀA → cột V
    #  1d  Singular values σᵢ = √λᵢ → ma trận Σ
    #  1e  Left-singular vectors u_i = Av_i/σ_i → ma trận U
    # ══════════════════════════════════════════════════════════════
    def beat_1_algebraic_svd(self):
 
        # ── Màn che + tiêu đề ──────────────────────────────────
        self.play(FadeOut(self.title_grp), run_time=0.8)
        veil = FullScreenRectangle().set_fill(BLACK, opacity=0.92).set_z_index(5)
        self.play(FadeIn(veil), run_time=0.8)
 
        title_alg = create_text_with_bg("Tính toán SVD từng bước",
                                        font_size=24, color=YELLOW,
                                        weight=BOLD).set_z_index(6)
        title_alg.to_edge(UP, buff=0.5)
        self.play(Write(title_alg))
 
        # ╔══════════════════════════════════════════════════════╗
        # ║  BƯỚC 1a — Tính AᵀA                                 ║
        # ╚══════════════════════════════════════════════════════╝
        self.show_subtitle("Bước 1a — Tính AᵀA (đây là ma trận đối xứng, nửa xác định dương).")
 
        eq_ATA = self.z6_tex(
            r"A^\top A = \begin{bmatrix}3&0\\1&2\end{bmatrix}"
            r"\begin{bmatrix}3&1\\0&2\end{bmatrix}"
            r"= \begin{bmatrix}9&3\\3&5\end{bmatrix}",
            font_size=36
        ).move_to(UP * 1.1)
        self.play(Write(eq_ATA), run_time=2.0)
        self.wait(1.5)
 
        # Lưu AᵀA vào góc, nhường chỗ tiếp tục
        ata_storage = eq_ATA.copy().set_z_index(6)
        self.play(
            FadeOut(eq_ATA),
            ata_storage.animate.scale(0.55).to_corner(UL, buff=0.5),
            run_time=1.0
        )
 
        # ╔══════════════════════════════════════════════════════╗
        # ║  BƯỚC 1b — Phương trình đặc trưng → λ₁, λ₂          ║
        # ╚══════════════════════════════════════════════════════╝
        self.show_subtitle("Bước 1b — Giải phương trình đặc trưng det(AᵀA − λI) = 0.")
 
        eq_det = self.z6_tex(
            r"\det\!\begin{bmatrix}9-\lambda&3\\3&5-\lambda\end{bmatrix}=0",
            font_size=36
        ).move_to(UP * 1.1)
        self.play(Write(eq_det), run_time=1.5)
        self.wait(1.0)
 
        eq_expand = self.z6_tex(
            r"(9-\lambda)(5-\lambda)-9=0"
            r"\;\Longrightarrow\;\lambda^2-14\lambda+36=0",
            font_size=33
        ).move_to(UP * 1.1)
        self.play(Transform(eq_det, eq_expand), run_time=1.2)
        self.wait(1.0)
 
        # Công thức nghiệm — giữ số đẹp dạng 7 ± √13
        eq_disc = self.z6_tex(
            r"\Delta = 14^2 - 4\cdot36 = 196-144 = 52 = 4\cdot13",
            font_size=30
        ).next_to(eq_expand, DOWN, buff=0.5)
        self.play(FadeIn(eq_disc, shift=UP * 0.2), run_time=1.0)
        self.wait(0.8)
 
        eq_roots = self.z6_tex(
            r"\lambda_1 = 7+\sqrt{13}\approx10{.}606\,,\quad"
            r"\lambda_2 = 7-\sqrt{13}\approx 3{.}394",
            font_size=32, color=YELLOW
        ).next_to(eq_disc, DOWN, buff=0.45)
        self.play(FadeIn(eq_roots, shift=UP * 0.2), run_time=1.0)
        self.play(Indicate(eq_roots))
        self.wait(1.5)
 
        # Lưu eigenvalues vào góc phải
        lam_storage = eq_roots.copy().set_z_index(6)
        self.play(
            FadeOut(eq_det), FadeOut(eq_disc), FadeOut(eq_roots),
            lam_storage.animate.scale(0.55).to_corner(UR, buff=0.5),
            run_time=1.0
        )
 
        # ╔══════════════════════════════════════════════════════╗
        # ║  BƯỚC 1c — Eigenvectors của AᵀA → cột của V         ║
        # ╚══════════════════════════════════════════════════════╝
        self.show_subtitle("Bước 1c — Tìm eigenvectors của AᵀA; các vector này sẽ là CỘT của V.")
 
        # ── λ₁ ──
        # Sửa lỗi Unicode: Dùng MarkupText cho chữ "Với"
        txt_voi = MarkupText("Với", font="Cambria", font_size=22, color=ORANGE).set_z_index(6)
        math_l1 = MathTex(r"\lambda_1 = 7+\sqrt{13}:", font_size=32, color=ORANGE).set_z_index(6)
        head_v1 = VGroup(txt_voi, math_l1).arrange(RIGHT, buff=0.15).move_to(UP * 1.5)

        # Trình bày hệ phương trình chi tiết hơn một chút
        sys_v1 = self.z6_tex(
            r"(A^\top A - \lambda_1 I)\mathbf{v} = \mathbf{0}",
            r"\Rightarrow \begin{bmatrix} 2-\sqrt{13} & 3 \\ 3 & -2-\sqrt{13} \end{bmatrix}",
            r"\begin{bmatrix} v_1 \\ v_2 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}",
            font_size=30
        ).next_to(head_v1, DOWN, buff=0.5)

        self.play(FadeIn(head_v1))
        self.play(Write(sys_v1[0])) # Hiện công thức tổng quát trước
        self.wait(0.5)
        self.play(FadeIn(sys_v1[1:3], shift=UP*0.2)) # Hiện ma trận số sau
 
        # Từ hàng 1: (2−√13)v₁ + 3v₂ = 0  →  v₁/v₂ = 3/(√13−2)
        step_v1 = self.z6_tex(
            r"\text{Hàng 1: }\;(2-\sqrt{13})\,v_1+3\,v_2=0"
            r"\;\Rightarrow\;"
            r"\mathbf{v}_1 \propto \begin{bmatrix}3\\\sqrt{13}-2\end{bmatrix}",
            font_size=27
        ).next_to(sys_v1, DOWN, buff=0.35)
        self.play(FadeIn(step_v1, shift=UP * 0.15), run_time=1.2)
        self.wait(0.8)
 
        # Chuẩn hóa
        norm_v1 = self.z6_tex(
            r"\|\mathbf{v}_1\|=\sqrt{9+({\sqrt{13}-2})^2}=\sqrt{26-4\sqrt{13}}"
            r"\approx 3{.}403",
            font_size=26
        ).next_to(step_v1, DOWN, buff=0.3)
        self.play(FadeIn(norm_v1, shift=UP * 0.15), run_time=1.0)
        self.wait(0.5)
 
        result_v1 = self.z6_tex(
            r"v_1 \approx \begin{bmatrix}0{.}882\\0{.}472\end{bmatrix}",
            font_size=32, color=ORANGE
        ).next_to(norm_v1, DOWN, buff=0.35)
        self.play(FadeIn(result_v1, shift=UP * 0.15), run_time=0.8)
        self.play(Indicate(result_v1, color=ORANGE))
        self.wait(1.0)
 
        # Lưu v1 và dọn màn
        v1_storage = result_v1.copy().set_z_index(6)
        self.play(
            FadeOut(head_v1), FadeOut(sys_v1),
            FadeOut(step_v1), FadeOut(norm_v1), FadeOut(result_v1),
            v1_storage.animate.scale(0.7).move_to(LEFT * 3.8 + DOWN * 1.0),
            run_time=1.0
        )
 
        # ── λ₂ ──
        self.show_subtitle("Bước 1c (tiếp) — Tìm eigenvector tương ứng với λ₂.")

        # FIX LỖI UNICODE: Tách chữ "Với" dùng MarkupText
        txt_voi_2 = MarkupText("Với", font="Cambria", font_size=22, color=PURPLE_B).set_z_index(6)
        math_l2 = MathTex(r"\lambda_2 = 7-\sqrt{13}:", font_size=32, color=PURPLE_B).set_z_index(6)
        head_v2 = VGroup(txt_voi_2, math_l2).arrange(RIGHT, buff=0.15).move_to(UP * 1.4)

        sys_v2 = self.z6_tex(
            r"(A^\top A - \lambda_2 I)\mathbf{v} = \mathbf{0}",
            r"\Rightarrow \begin{bmatrix} 2+\sqrt{13} & 3 \\ 3 & \sqrt{13}-2 \end{bmatrix}",
            r"\begin{bmatrix} v_1 \\ v_2 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}",
            font_size=27
        ).next_to(head_v2, DOWN, buff=0.35)

        self.play(FadeIn(head_v2), Write(sys_v2))
        self.wait(1.0)
 
        # --- BƯỚC CHUẨN HÓA v2 ---
        v2_prop = self.z6_tex(r"\mathbf{v}_2 \propto \begin{bmatrix}3\\-(2+\sqrt{13})\end{bmatrix}", font_size=27)
        
        # Tạo mũi tên và nhãn tiếng Việt riêng (MarkupText không lỗi Unicode)
        arrow_tex = MathTex(r"\xrightarrow{\hspace{1.5cm}}", font_size=27).set_z_index(6)
        label_chuan_hoa = MarkupText("chuẩn hóa", font="Cambria", font_size=14).set_z_index(6)
        label_chuan_hoa.next_to(arrow_tex, UP, buff=0.05)
        arrow_group = VGroup(arrow_tex, label_chuan_hoa)
        
        v2_approx = self.z6_tex(r"v_2 \approx \begin{bmatrix}0.472\\-0.882\end{bmatrix}", font_size=27)

        # Gom tất cả thành 1 dòng duy nhất
        step_v2 = VGroup(v2_prop, arrow_group, v2_approx).arrange(RIGHT, buff=0.2).next_to(sys_v2, DOWN, buff=0.4)
        
        self.play(FadeIn(step_v2, shift=UP * 0.15), run_time=1.2)
        self.wait(0.8)

        # Kết quả v2 nhấn mạnh
        result_v2 = self.z6_tex(
            r"v_2 \approx \begin{bmatrix}0.472\\-0.882\end{bmatrix}",
            font_size=32, color=PURPLE_B
        ).next_to(step_v2, DOWN, buff=0.35)
        
        self.play(FadeIn(result_v2, shift=UP * 0.15), run_time=0.8)
        self.play(Indicate(result_v2, color=PURPLE_B))
        self.wait(0.8)
 
        # Ghép thành V
        self.show_subtitle("Ghép v₁, v₂ theo cột → ma trận V (và lấy chuyển vị → Vᵀ).")
 
        v2_storage = result_v2.copy().set_z_index(6)
        self.play(
            FadeOut(head_v2), FadeOut(sys_v2),
            FadeOut(step_v2), FadeOut(result_v2),
            v2_storage.animate.scale(0.7).move_to(RIGHT * 3.8 + DOWN * 1.0),
            run_time=1.0
        )
 
        # Mũi tên + bằng V
        arrow_V = Arrow(LEFT * 1.5 + DOWN * 1.0, RIGHT * 1.5 + DOWN * 1.0,
                        color=WHITE, buff=0, stroke_width=3, tip_length=0.22).set_z_index(6)
        lbl_arrange = self.z6_tex(
            r"V = [v_1 \mid v_2]"
            r"\approx\begin{bmatrix}0{.}882&0{.}472\\0{.}472&-0{.}882\end{bmatrix}",
            font_size=30, color=TEAL
        ).move_to(DOWN * 1.0)
 
        self.play(GrowArrow(arrow_V), run_time=0.6)
        self.play(
            FadeOut(v1_storage), FadeOut(v2_storage), FadeOut(arrow_V),
            FadeIn(lbl_arrange, shift=UP * 0.2),
            run_time=1.0
        )
        self.wait(0.8)
 
        # Vᵀ
        lbl_Vt = self.z6_tex(
            r"V^\top \approx \begin{bmatrix}0{.}882&0{.}472\\0{.}472&-0{.}882\end{bmatrix}^\top"
            r"= \begin{bmatrix}0{.}882&0{.}472\\0{.}472&-0{.}882\end{bmatrix}",
            font_size=28, color=TEAL
        ).next_to(lbl_arrange, DOWN, buff=0.3)
        self.play(FadeIn(lbl_Vt, shift=UP * 0.15), run_time=1.0)
        self.wait(1.0)
 
        # Lưu V vào góc
        V_storage = VGroup(lbl_arrange, lbl_Vt).copy().set_z_index(6)
        self.play(
            FadeOut(lbl_arrange), FadeOut(lbl_Vt),
            V_storage.animate.scale(0.5).to_corner(DL, buff=0.4),
            run_time=1.0
        )
 
        # ╔══════════════════════════════════════════════════════╗
        # ║  BƯỚC 1d — Singular values σᵢ = √λᵢ → ma trận Σ    ║
        # ╚══════════════════════════════════════════════════════╝
        self.show_subtitle("Bước 1d — Singular values: σᵢ = √λᵢ. Xếp giảm dần vào đường chéo Σ.")
 
        sigma_steps = self.z6_tex(
            r"\sigma_1 = \sqrt{\lambda_1} = \sqrt{7+\sqrt{13}} \approx 3{.}257"
            r"\qquad"
            r"\sigma_2 = \sqrt{\lambda_2} = \sqrt{7-\sqrt{13}} \approx 1{.}842",
            font_size=30, color=GREEN_B
        ).move_to(UP * 0.8)
        self.play(Write(sigma_steps), run_time=1.8)
        self.wait(1.0)
 
        sigma_mat = self.z6_tex(
            r"\Sigma = \begin{bmatrix}\sigma_1 & 0 \\ 0 & \sigma_2\end{bmatrix}"
            r"\approx \begin{bmatrix}3{.}257 & 0 \\ 0 & 1{.}842\end{bmatrix}",
            font_size=34, color=GREEN_B
        ).next_to(sigma_steps, DOWN, buff=0.5)
        self.play(FadeIn(sigma_mat, shift=UP * 0.2), run_time=1.0)
        self.play(Indicate(sigma_mat, color=GREEN_B))
        self.wait(1.2)
 
        Sig_storage = sigma_mat.copy().set_z_index(6)
        self.play(
            FadeOut(sigma_steps), FadeOut(sigma_mat),
            Sig_storage.animate.scale(0.55).to_corner(DR, buff=0.4),
            run_time=1.0
        )
 
        # ╔══════════════════════════════════════════════════════╗
        # ║  BƯỚC 1e — u_i = A v_i / σ_i → ma trận U           ║
        # ╚══════════════════════════════════════════════════════╝
        self.show_subtitle("Bước 1e — Tính left-singular vectors: u_i = A·v_i / σ_i.")
 
        # Công thức tổng quát
        formula_u = self.z6_tex(
            r"u_i = \frac{A\,v_i}{\sigma_i}",
            font_size=42
        ).move_to(UP * 1.2)
        self.play(Write(formula_u), run_time=1.2)
        self.wait(0.8)
 
        # ── Tính u₁ ──
        calc_u1_a = self.z6_tex(
            r"A\,v_1 = \begin{bmatrix}3&1\\0&2\end{bmatrix}"
            r"\begin{bmatrix}0{.}882\\0{.}472\end{bmatrix}"
            r"= \begin{bmatrix}3{.}118\\0{.}944\end{bmatrix}",
            font_size=28
        ).next_to(formula_u, DOWN, buff=0.4)
        self.play(FadeIn(calc_u1_a, shift=UP * 0.15), run_time=1.5)
        self.wait(0.8)
 
        calc_u1_b = self.z6_tex(
            r"u_1 = \frac{1}{3{.}257}\begin{bmatrix}3{.}118\\0{.}944\end{bmatrix}"
            r"\approx \begin{bmatrix}0{.}957\\0{.}290\end{bmatrix}",
            font_size=28, color=RED_B
        ).next_to(calc_u1_a, DOWN, buff=0.35)
        self.play(FadeIn(calc_u1_b, shift=UP * 0.15), run_time=1.2)
        self.play(Indicate(calc_u1_b, color=RED_B))
        self.wait(0.8)
 
        u1_storage = calc_u1_b.copy().set_z_index(6)
        self.play(
            FadeOut(calc_u1_a), FadeOut(calc_u1_b),
            u1_storage.animate.scale(0.65).move_to(LEFT * 3.5 + DOWN * 0.5),
            run_time=1.0
        )
 
        # ── Tính u₂ ──
        self.show_subtitle("Tương tự, tính u₂ = A·v₂ / σ₂.")
 
        calc_u2_a = self.z6_tex(
            r"A\,v_2 = \begin{bmatrix}3&1\\0&2\end{bmatrix}"
            r"\begin{bmatrix}0{.}472\\-0{.}882\end{bmatrix}"
            r"= \begin{bmatrix}0{.}534\\-1{.}764\end{bmatrix}",
            font_size=28
        ).next_to(formula_u, DOWN, buff=0.4)
        self.play(FadeIn(calc_u2_a, shift=UP * 0.15), run_time=1.5)
        self.wait(0.8)
 
        calc_u2_b = self.z6_tex(
            r"u_2 = \frac{1}{1{.}842}\begin{bmatrix}0{.}534\\-1{.}764\end{bmatrix}"
            r"\approx \begin{bmatrix}0{.}290\\-0{.}957\end{bmatrix}",
            font_size=28, color=RED_B
        ).next_to(calc_u2_a, DOWN, buff=0.35)
        self.play(FadeIn(calc_u2_b, shift=UP * 0.15), run_time=1.2)
        self.play(Indicate(calc_u2_b, color=RED_B))
        self.wait(0.8)
 
        u2_storage = calc_u2_b.copy().set_z_index(6)
        self.play(
            FadeOut(calc_u2_a), FadeOut(calc_u2_b),
            u2_storage.animate.scale(0.65).move_to(RIGHT * 3.5 + DOWN * 0.5),
            run_time=1.0
        )
 
        # Ghép thành U
        self.show_subtitle("Ghép u₁, u₂ theo cột → ma trận trực giao U.")
 
        arrow_U = Arrow(LEFT * 1.5 + DOWN * 0.5, RIGHT * 1.5 + DOWN * 0.5,
                        color=WHITE, buff=0, stroke_width=3, tip_length=0.22).set_z_index(6)
        lbl_U_result = self.z6_tex(
            r"U = [u_1 \mid u_2]"
            r"\approx\begin{bmatrix}0{.}957&0{.}290\\0{.}290&-0{.}957\end{bmatrix}",
            font_size=30, color=RED_B
        ).move_to(DOWN * 0.5)
 
        self.play(GrowArrow(arrow_U), run_time=0.5)
        self.play(
            FadeOut(u1_storage), FadeOut(u2_storage), FadeOut(arrow_U),
            FadeIn(lbl_U_result, shift=UP * 0.2),
            run_time=1.0
        )
        self.wait(0.8)
 
        # --- KIỂM TRA TÍNH TRỰC GIAO CỦA U ---
        # 1. Tạo phần toán học (MathTex - an toàn)
        check_math = self.z6_tex(r"U^\top U \approx I \quad\checkmark", font_size=28, color=GREEN_C)
        
        # 2. Tạo phần chú thích tiếng Việt (MarkupText - hỗ trợ Unicode tốt)
        check_text = MarkupText("(trực giao!)", font="Cambria", font_size=18, color=GREEN_C).set_z_index(6)
        
        # 3. Gom nhóm và căn chỉnh
        check_orth = VGroup(check_math, check_text).arrange(RIGHT, buff=0.2)
        check_orth.next_to(lbl_U_result, DOWN, buff=0.3)
        
        self.play(FadeIn(check_orth, shift=UP * 0.15), run_time=0.8)
        self.wait(1.0)
 
        U_storage = VGroup(lbl_U_result, check_orth).copy().set_z_index(6)
        self.play(
            FadeOut(lbl_U_result), FadeOut(check_orth), FadeOut(formula_u),
            # THAY to_corner(DR) BẰNG DÒNG DƯỚI ĐÂY:
            U_storage.animate.scale(0.5).next_to(Sig_storage, UP, buff=0.5).to_edge(RIGHT, buff=0.4),
            run_time=1.0
        )
 
        # ── Kết quả tổng hợp hoàn chỉnh ──
        self.show_subtitle("Kết hợp lại: A = U Σ Vᵀ — phân rã SVD hoàn chỉnh!")
 
        eq_final = self.z6_tex(
            r"A = \underbrace{\begin{bmatrix}0{.}957&0{.}290\\0{.}290&-0{.}957\end{bmatrix}}_{U}"
            r"\underbrace{\begin{bmatrix}3{.}257&0\\0&1{.}842\end{bmatrix}}_{\Sigma}"
            r"\underbrace{\begin{bmatrix}0{.}882&0{.}472\\0{.}472&-0{.}882\end{bmatrix}}_{V^\top}",
            font_size=24
        ).move_to(UP * 0.5)
        self.play(FadeIn(eq_final, shift=UP * 0.2), run_time=2.0)
        box_final = SurroundingRectangle(eq_final, color=YELLOW, buff=0.2, stroke_width=2)
        box_final.set_z_index(6)
        self.play(Create(box_final), run_time=0.8)
        self.wait(2.5)
 
        # Dọn sạch trước khi sang beat hình học
        self.play(
            FadeOut(title_alg), FadeOut(eq_final), FadeOut(box_final),
            FadeOut(ata_storage), FadeOut(lam_storage),
            FadeOut(V_storage), FadeOut(Sig_storage), FadeOut(U_storage),
            FadeOut(veil),
            run_time=1.2
        )

    # ══════════════════════════════════════════════════════════════
    #  BEAT 2 — Hình học: Bước V^T (xoay đầu tiên)
    # ══════════════════════════════════════════════════════════════
    def beat_2_geometric_v_rotation(self):
        """Áp dụng V^T lên hình tròn đơn vị + các vector cơ sở."""
        self.show_subtitle("Phần 2 – Trực quan hóa SVD như 3 phép biến đổi liên tiếp.", run_time=1.5)
        self.wait(0.5)

        # Vẽ lại plane sạch
        self.plane = self.fresh_plane()
        self.play(FadeIn(self.plane), run_time=0.5)

        # ─ Hình tròn đơn vị & axes ─
        self.circle = Circle(radius=1.0, color=BLUE_B, stroke_width=2.5)
        e1 = np.array([1.0, 0.0, 0.0])
        e2 = np.array([0.0, 1.0, 0.0])
        kw = dict(buff=0, stroke_width=5, tip_length=0.25)
        self.ax1 = Arrow(ORIGIN, e1, color=RED_C, **kw)
        self.ax2 = Arrow(ORIGIN, e2, color=GREEN_C, **kw)
        self.lbl1 = MathTex(r"\hat{e}_1", font_size=28, color=RED_C).next_to(e1, DR, buff=0.05)
        self.lbl2 = MathTex(r"\hat{e}_2", font_size=28, color=GREEN_C).next_to(e2, UL, buff=0.05)

        step_lbl = create_text_with_bg("Bước 1: Áp dụng Vᵀ — xoay hệ tọa độ về hướng V",
                                       font_size=20, color=TEAL_B)
        step_lbl.to_edge(UP, buff=0.85)

        self.play(
            Create(self.circle),
            GrowArrow(self.ax1), GrowArrow(self.ax2),
            Write(self.lbl1), Write(self.lbl2),
            FadeIn(step_lbl),
            run_time=1.5
        )
        self.show_subtitle("Vᵀ xoay không gian đầu vào sao cho các right-singular vectors thẳng hàng với trục.", run_time=2.0)
        self.wait(0.8)

        # Áp dụng Vᵀ
        self.lbl1.add_updater(lambda m: m.next_to(self.ax1.get_end(), DR, buff=0.05))
        self.lbl2.add_updater(lambda m: m.next_to(self.ax2.get_end(), UL, buff=0.05))

        self.play(
            ApplyMatrix(Vt_3x3, self.circle),
            ApplyMatrix(Vt_3x3, self.ax1),
            ApplyMatrix(Vt_3x3, self.ax2),
            ApplyMatrix(Vt_3x3, self.plane),
            run_time=3.0, rate_func=smooth
        )
        self.lbl1.clear_updaters()
        self.lbl2.clear_updaters()

        self.show_subtitle("Hình tròn đơn vị vẫn là hình tròn — Vᵀ là phép xoay (isometry).", run_time=2.0)
        self.wait(1.5)
        self.play(FadeOut(step_lbl), run_time=0.5)

    # ══════════════════════════════════════════════════════════════
    #  BEAT 3 — Hình học: Bước Σ (co giãn)
    # ══════════════════════════════════════════════════════════════
    def beat_3_geometric_sigma_scale(self):
        step_lbl = create_text_with_bg(
            "Bước 2: Áp dụng Σ — co giãn theo trục tọa độ",
            font_size=20, color=YELLOW)
        step_lbl.to_edge(UP, buff=0.85)
        self.play(FadeIn(step_lbl), run_time=0.8)

        self.show_subtitle("Σ kéo dài theo chiều ngang (σ₁ ≈ 3.16) và chiều dọc (σ₂ ≈ 1.84).", run_time=2.0)
        self.wait(0.5)

        # Nhãn sigma hiển thị trước khi biến đổi
        sig_lbl1 = MathTex(r"\sigma_1 \approx 3.162", font_size=26, color=YELLOW)
        sig_lbl1.move_to(RIGHT * 1.8 + UP * 0.3)
        sig_lbl1_bg = BackgroundRectangle(sig_lbl1, color=BLACK, fill_opacity=0.8, buff=0.1)
        sig1_grp = VGroup(sig_lbl1_bg, sig_lbl1)

        sig_lbl2 = MathTex(r"\sigma_2 \approx 1.842", font_size=26, color=YELLOW)
        sig_lbl2.move_to(LEFT * 0.5 + UP * 1.3)
        sig_lbl2_bg = BackgroundRectangle(sig_lbl2, color=BLACK, fill_opacity=0.8, buff=0.1)
        sig2_grp = VGroup(sig_lbl2_bg, sig_lbl2)

        self.play(FadeIn(sig1_grp), FadeIn(sig2_grp), run_time=0.8)

        self.lbl1.add_updater(lambda m: m.next_to(self.ax1.get_end(), DR, buff=0.05))
        self.lbl2.add_updater(lambda m: m.next_to(self.ax2.get_end(), UL, buff=0.05))

        self.play(
            ApplyMatrix(Sig_3x3, self.circle),
            ApplyMatrix(Sig_3x3, self.ax1),
            ApplyMatrix(Sig_3x3, self.ax2),
            ApplyMatrix(Sig_3x3, self.plane),
            run_time=3.5, rate_func=smooth
        )
        self.lbl1.clear_updaters()
        self.lbl2.clear_updaters()

        self.show_subtitle("Hình tròn biến thành hình ELLIPSE — đây là đặc trưng của biến đổi tuyến tính!", run_time=2.0)
        self.wait(1.5)
        self.play(FadeOut(step_lbl), FadeOut(sig1_grp), FadeOut(sig2_grp), run_time=0.5)

    # ══════════════════════════════════════════════════════════════
    #  BEAT 4 — Hình học: Bước U (xoay cuối)
    # ══════════════════════════════════════════════════════════════
    def beat_4_geometric_u_rotation(self):
        step_lbl = create_text_with_bg(
            "Bước 3: Áp dụng U — xoay về vị trí cuối cùng",
            font_size=20, color=RED_B)
        step_lbl.to_edge(UP, buff=0.85)
        self.play(FadeIn(step_lbl), run_time=0.8)

        self.show_subtitle("U xoay ellipse đến vị trí đầu ra cuối cùng — left-singular vectors.", run_time=2.0)
        self.wait(0.5)

        self.lbl1.add_updater(lambda m: m.next_to(self.ax1.get_end(), DR, buff=0.05))
        self.lbl2.add_updater(lambda m: m.next_to(self.ax2.get_end(), UL, buff=0.05))

        self.play(
            ApplyMatrix(U_3x3, self.circle),
            ApplyMatrix(U_3x3, self.ax1),
            ApplyMatrix(U_3x3, self.ax2),
            ApplyMatrix(U_3x3, self.plane),
            run_time=3.0, rate_func=smooth
        )
        self.lbl1.clear_updaters()
        self.lbl2.clear_updaters()

        self.show_subtitle("Kết quả: Ellipse cuối cùng = ảnh của hình tròn đơn vị qua A!", run_time=2.0)
        self.wait(1.5)

        # Tô màu ellipse nổi bật
        self.play(
            self.circle.animate.set_fill(BLUE, opacity=0.18).set_stroke(BLUE_B, width=3),
            run_time=1.0
        )
        self.play(FadeOut(step_lbl), run_time=0.5)

    # ══════════════════════════════════════════════════════════════
    #  BEAT 5 — Toàn cảnh: Hình tròn → A → Ellipse (song song)
    # ══════════════════════════════════════════════════════════════
    def beat_5_unit_circle_full(self):
        """Làm sạch và hiển thị so sánh trước/sau trên cùng màn hình."""
        self.show_subtitle("Tổng kết hình học: Nhìn toàn cảnh biến đổi A = U Σ Vᵀ.", run_time=1.5)

        # Xóa tất cả để vẽ lại so sánh
        self.play(
            FadeOut(self.plane), FadeOut(self.circle),
            FadeOut(self.ax1), FadeOut(self.ax2),
            FadeOut(self.lbl1), FadeOut(self.lbl2),
            run_time=1.0
        )

        # Plane mới chia đôi
        plane_l = NumberPlane(
            x_range=[-4, 4], y_range=[-3, 3],
            background_line_style={"stroke_color": GREY, "stroke_width": 0.8, "stroke_opacity": 0.3},
            axis_config={"stroke_color": GREY_B, "stroke_width": 1.2},
        ).scale(0.85).shift(LEFT * 3.3)
        plane_r = NumberPlane(
            x_range=[-4, 4], y_range=[-3, 3],
            background_line_style={"stroke_color": GREY, "stroke_width": 0.8, "stroke_opacity": 0.3},
            axis_config={"stroke_color": GREY_B, "stroke_width": 1.2},
        ).scale(0.85).shift(RIGHT * 3.3)

        # Hình tròn đơn vị (bên trái)
        circle_l = Circle(radius=0.85, color=BLUE_B, stroke_width=2.5).shift(LEFT * 3.3)
        circle_l.set_fill(BLUE, opacity=0.12)

        # Ellipse sau biến đổi (bên phải) — tính thực tế
        # Các điểm trên đường tròn sau biến đổi A
        theta_vals = np.linspace(0, 2 * np.pi, 200)
        pts = np.array([[np.cos(t), np.sin(t)] for t in theta_vals])
        transformed_pts = (A_mat @ pts.T).T  # shape (200, 2)
        # Vẽ bằng VMobject
        ellipse_r = VMobject(color=ORANGE, stroke_width=2.5, fill_color=ORANGE, fill_opacity=0.12)
        path_pts = [np.array([p[0], p[1], 0]) * 0.85 + RIGHT * 3.3 for p in transformed_pts]
        ellipse_r.set_points_as_corners(path_pts)
        ellipse_r.make_smooth()

        # Nhãn
        lbl_input = create_text_with_bg("Đầu vào\n(Hình tròn đơn vị)", font_size=18, color=BLUE_B)
        lbl_input.next_to(plane_l, UP, buff=0.2)
        lbl_output = create_text_with_bg("Đầu ra = A · (hình tròn)\n(Ellipse)", font_size=18, color=ORANGE)
        lbl_output.next_to(plane_r, UP, buff=0.2)

        # Mũi tên ở giữa
        arrow_mid = Arrow(LEFT * 0.5, RIGHT * 0.5, color=WHITE,
                          buff=0, stroke_width=4, tip_length=0.3)
        lbl_A_mid = MathTex(r"A = U\Sigma V^\top", font_size=26, color=WHITE)
        lbl_A_mid.next_to(arrow_mid, UP, buff=0.1)

        self.play(
            FadeIn(plane_l), FadeIn(plane_r),
            Create(circle_l),
            FadeIn(lbl_input), FadeIn(lbl_output),
            GrowArrow(arrow_mid), Write(lbl_A_mid),
            run_time=1.5
        )
        self.wait(0.8)
        self.play(Create(ellipse_r), run_time=2.0, rate_func=smooth)
        self.wait(1.0)

        # Vẽ bán trục lớn và nhỏ của ellipse
        # Hướng left-singular vector u1
        u1 = U_np[:, 0] * S_np[0] * 0.85
        u2 = U_np[:, 1] * S_np[1] * 0.85
        ax_major = Arrow(RIGHT * 3.3, RIGHT * 3.3 + np.array([u1[0], u1[1], 0]),
                         color=GREEN_C, buff=0, stroke_width=4, tip_length=0.22)
        ax_minor = Arrow(RIGHT * 3.3, RIGHT * 3.3 + np.array([u2[0], u2[1], 0]),
                         color=YELLOW, buff=0, stroke_width=4, tip_length=0.22)
        lbl_ax1 = MathTex(r"\sigma_1 u_1", font_size=22, color=GREEN_C)
        lbl_ax1.next_to(ax_major.get_end(), UR, buff=0.1)
        lbl_ax2 = MathTex(r"\sigma_2 u_2", font_size=22, color=YELLOW)
        lbl_ax2.next_to(ax_minor.get_end(), UL, buff=0.1)

        self.show_subtitle("Các bán trục của ellipse = σᵢ · uᵢ: singular values × left-singular vectors.", run_time=2.0)
        self.play(GrowArrow(ax_major), GrowArrow(ax_minor),
                  Write(lbl_ax1), Write(lbl_ax2), run_time=1.5)
        self.wait(2.0)

        self.play(
            FadeOut(plane_l), FadeOut(plane_r),
            FadeOut(circle_l), FadeOut(ellipse_r),
            FadeOut(lbl_input), FadeOut(lbl_output),
            FadeOut(arrow_mid), FadeOut(lbl_A_mid),
            FadeOut(ax_major), FadeOut(ax_minor),
            FadeOut(lbl_ax1), FadeOut(lbl_ax2),
            run_time=1.0
        )

    # ══════════════════════════════════════════════════════════════
    #  BEAT 6 — Tổng kết: Hiển thị A = U Σ Vᵀ đầy đủ
    # ══════════════════════════════════════════════════════════════
    def beat_6_summary(self):
        self.clear_subtitle()
        self.show_subtitle("Kết quả phân rã SVD hoàn chỉnh của A.", run_time=1.2)

        # Tiêu đề
        title_sum = create_text_with_bg("Phân rã SVD hoàn chỉnh: A = U Σ Vᵀ",
                                        font_size=24, color=YELLOW, weight=BOLD)
        title_sum.to_edge(UP, buff=0.45)
        self.play(Write(title_sum))

        # ─ Ma trận A ─
        lbl_A = MathTex("A", font_size=34, color=ORANGE)
        mat_A = MathTex(r"\begin{bmatrix} 3 & 1 \\ 0 & 2 \end{bmatrix}",
                        font_size=38, color=ORANGE)
        grp_A = VGroup(lbl_A, mat_A).arrange(DOWN, buff=0.15)

        eq = MathTex("=", font_size=42, color=WHITE)

        # ─ Ma trận U ─
        lbl_U = MathTex("U", font_size=34, color=RED_B)
        mat_U = MathTex(
            r"\begin{bmatrix} -0.981 & -0.196 \\ -0.196 & 0.981 \end{bmatrix}",
            font_size=28, color=RED_B)
        grp_U = VGroup(lbl_U, mat_U).arrange(DOWN, buff=0.15)

        # ─ Ma trận Σ ─
        lbl_S = MathTex(r"\Sigma", font_size=34, color=GREEN_B)
        mat_S = MathTex(
            r"\begin{bmatrix} 3.162 & 0 \\ 0 & 1.842 \end{bmatrix}",
            font_size=28, color=GREEN_B)
        grp_S = VGroup(lbl_S, mat_S).arrange(DOWN, buff=0.15)

        # ─ Ma trận Vᵀ ─
        lbl_Vt = MathTex("V^\\top", font_size=34, color=TEAL)
        mat_Vt = MathTex(
            r"\begin{bmatrix} -0.981 & -0.196 \\ 0.196 & -0.981 \end{bmatrix}",
            font_size=28, color=TEAL)
        grp_Vt = VGroup(lbl_Vt, mat_Vt).arrange(DOWN, buff=0.15)

        full = VGroup(grp_A, eq, grp_U, grp_S, grp_Vt).arrange(RIGHT, buff=0.3)
        full.move_to(ORIGIN + DOWN * 0.3)

        self.play(FadeIn(grp_A), run_time=0.8)
        self.play(Write(eq), run_time=0.4)
        self.show_subtitle("U là ma trận trực giao chứa left-singular vectors.", run_time=1.5)
        self.play(FadeIn(grp_U, shift=LEFT * 0.2), run_time=1.0)
        self.wait(0.5)
        self.show_subtitle("Σ là ma trận đường chéo chứa singular values σ₁ ≥ σ₂ > 0.", run_time=1.5)
        self.play(FadeIn(grp_S, shift=LEFT * 0.2), run_time=1.0)
        self.wait(0.5)
        self.show_subtitle("Vᵀ là ma trận trực giao chứa right-singular vectors (transposed).", run_time=1.5)
        self.play(FadeIn(grp_Vt, shift=LEFT * 0.2), run_time=1.0)
        self.wait(1.5)

        # Hộp highlight toàn bộ
        box_full = SurroundingRectangle(full, color=YELLOW, buff=0.2, stroke_width=2.5)
        self.play(Create(box_full), run_time=1.0)
        self.show_subtitle("SVD luôn tồn tại và duy nhất (với σᵢ > 0) cho MỌI ma trận thực!", run_time=2.5)
        self.wait(2.0)

        # Ghi chú tính chất quan trọng
        # 1. Tạo các dòng tính chất bằng cách kết hợp Math (công thức) và Text (tiếng Việt)
        # Dòng 1: Tính trực giao
        line1_math = MathTex(r"\bullet\; U^\top U = I,\quad V^\top V = I", font_size=22, color=GREY_A)
        line1_text = MarkupText("(trực giao)", font="Cambria", font_size=16, color=GREY_A)
        line1 = VGroup(line1_math, line1_text).arrange(RIGHT, buff=0.2)

        # Dòng 2: Giá trị suy biến
        line2_math = MathTex(r"\bullet\; \sigma_1 \geq \sigma_2 \geq 0", font_size=22, color=GREY_A)
        line2_text = MarkupText("(giá trị suy biến)", font="Cambria", font_size=16, color=GREY_A)
        line2 = VGroup(line2_math, line2_text).arrange(RIGHT, buff=0.2)

        # Gom các dòng lại thành một nhóm
        prop_lines = VGroup(line1, line2).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        
        # Đặt vị trí cho nhóm tổng kết (ví dụ ở góc dưới bên trái hoặc dưới công thức chính)
        prop_lines.to_corner(DL, buff=1.0)

        # Hiển thị
        self.play(FadeIn(prop_lines, shift=UP*0.2))
        self.wait(3)

        # Màn hình tối kết thúc
        veil = FullScreenRectangle().set_fill(BLACK, opacity=1.0).set_z_index(10)
        self.play(FadeIn(veil), run_time=2.0)
        self.wait(1.0)