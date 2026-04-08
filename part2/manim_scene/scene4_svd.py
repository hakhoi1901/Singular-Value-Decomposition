from manim import *
import numpy as np

# Ma trận A của đồ án
A_mat = np.array([[3.0, 1.0], 
                  [0.0, 2.0]])

# Ma trận 3D cho Manim
A_3x3 = np.array([[3, 1, 0],
                  [0, 2, 0],
                  [0, 0, 1]], dtype=float)

def create_text_with_bg(text, font_size=20, color=WHITE, bg_opacity=0.8, font="Cambria", weight=NORMAL):
    txt_obj = MarkupText(text, font=font, font_size=font_size, color=color, weight=weight).set_z_index(1)
    bg = BackgroundRectangle(txt_obj, color=BLACK, fill_opacity=bg_opacity, buff=0.15).set_z_index(0)
    return VGroup(bg, txt_obj)

class Scene4_Diagonalization(MovingCameraScene):
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

    def construct(self):
        self.beat_4_1_eigenvectors()
        self.beat_4_1_5_algebraic_calculation() # BƯỚC TÍNH TOÁN MỚI
        self.beat_4_2_eigen_transformation()
        self.beat_4_3_matrix_assembly() 
        self.beat_4_4_the_ultimate_comparison()

    def beat_4_1_eigenvectors(self):
        """Nhịp 4.1: Giới thiệu Chéo hóa & Bối cảnh"""
        self.plane = NumberPlane(
            x_range=[-10, 10, 1], y_range=[-6, 6, 1],
            background_line_style={"stroke_color": GREY, "stroke_width": 1, "stroke_opacity": 0.35},
            axis_config={"stroke_color": GREY_B, "stroke_width": 1.5},
        )
        self.add(self.plane)

        formula_diag = MathTex(r"\text{Diagonalization: }\quad A = P\,D\,P^{-1}", font_size=36, color=WHITE)
        formula_diag[0][16:17].set_color(ORANGE) # A
        formula_diag[0][18:19].set_color(TEAL)   # P
        formula_diag[0][19:20].set_color(YELLOW) # D
        formula_bg = BackgroundRectangle(formula_diag, color=BLACK, fill_opacity=0.8, buff=0.15)
        self.formula_grp = VGroup(formula_bg, formula_diag).to_edge(UP, buff=0.3)
        self.play(FadeIn(self.formula_grp, shift=DOWN*0.2), run_time=1.0)

        self.show_subtitle("Chuyển sang phương pháp Chéo hóa. Mục tiêu là tìm kiếm 'sự bất biến'.", run_time=1.5)
        self.wait(1.0)

    def beat_4_1_5_algebraic_calculation(self):
        """Nhịp 4.1.5: Giải phẫu Đại số - Từng bước một cách mượt mà"""
        # 1. Khởi tạo màn che và tiêu đề
        veil = FullScreenRectangle().set_fill(BLACK, opacity=0.9).set_z_index(5)
        self.play(FadeIn(veil), run_time=1.0)

        title = create_text_with_bg("Giải phẫu ma trận A", font_size=24, color=YELLOW).set_z_index(6)
        title.to_edge(UP, buff=0.5)
        self.play(Write(title))

        # --- PHẦN 1: TÌM EIGENVALUES (Tập trung ở giữa) ---
        self.show_subtitle("Bước 1: Tìm các Giá trị riêng bằng phương trình đặc trưng.", run_time=1.5)
        
        eq1 = MathTex(r"\det(A - \lambda I) = 0", font_size=42).set_z_index(6).move_to(UP*1)
        eq2 = MathTex(r"\det \begin{bmatrix} 3-\lambda & 1 \\ 0 & 2-\lambda \end{bmatrix} = 0", font_size=42).set_z_index(6).move_to(UP*1)
        eq3 = MathTex(r"(3-\lambda)(2-\lambda) - (1)(0) = 0", font_size=42).set_z_index(6).move_to(UP*1)
        
        self.play(Write(eq1))
        self.wait(1.3)
        self.play(Transform(eq1, eq2)) # Biến đổi mượt mà giữa các bước
        self.wait(1.3)
        self.play(Transform(eq1, eq3))
        self.wait(1.3)

        # Kết quả Lambda
        res_lambda = MathTex(r"\lambda_1 = 3, \quad \lambda_2 = 2", font_size=45, color=YELLOW).set_z_index(6)
        res_lambda.next_to(eq1, DOWN, buff=0.8)
        self.play(FadeIn(res_lambda, shift=UP*0.3))
        self.play(Indicate(res_lambda))
        self.wait(1.5)

        # THU GOM: Đẩy kết quả lambda vào một "hộp lưu trữ" ở góc để lấy chỗ giải tiếp
        lambda_box = SurroundingRectangle(res_lambda, color=GREY_B, buff=0.2).set_z_index(6)
        lambda_storage = VGroup(res_lambda, lambda_box)
        
        self.play(
            FadeOut(eq1),
            FadeOut(title),
            lambda_storage.animate.scale(0.6).to_corner(UL, buff=0.5),
            run_time=1.2
        )

        # --- PHẦN 2: TÌM EIGENVECTORS (Giải từng cái một) ---
        
        # --- Giải cho Lambda 1 ---
        self.show_subtitle("Với mỗi lambda, ta tìm một không gian vector riêng tương ứng.")
        v1_head = MathTex(r"\text{Xét } \lambda_1 = 3:", font_size=34, color=RED_B).set_z_index(6).to_edge(UP, buff=1.2)
        v1_sys = MathTex(r"\begin{bmatrix} 3-3 & 1 \\ 0 & 2-3 \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = \vec{0}", font_size=38).set_z_index(6)
        v1_simp = MathTex(r"\begin{bmatrix} 0 & 1 \\ 0 & -1 \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix} \implies y = 0", font_size=36).set_z_index(6)
        v1_final = MathTex(r"\Rightarrow \vec{v}_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}", font_size=42, color=RED_C).set_z_index(6)
        v1_final.next_to(v1_simp, DOWN, buff=0.5)

        self.play(Write(v1_head))
        self.play(FadeIn(v1_sys, shift=DOWN*0.2))
        self.wait(1)
        self.play(Transform(v1_sys, v1_simp))
        self.wait(1)
        self.play(Write(v1_final))
        self.play(Indicate(v1_final))
        self.wait(1)

        # Lưu vector 1 vào góc UR
        v1_storage = v1_final.copy().set_z_index(6)
        self.play(
            FadeOut(v1_head), FadeOut(v1_sys), FadeOut(v1_final),
            v1_storage.animate.scale(0.7).to_corner(UR, buff=0.5),
            run_time=1
        )
        
        # --- Giải cho Lambda 2 ---
        v2_head = MathTex(r"\text{Xét } \lambda_2 = 2:", font_size=34, color=BLUE_B).set_z_index(6).to_edge(UP, buff=1.2)
        v2_sys = MathTex(r"\begin{bmatrix} 3-2 & 1 \\ 0 & 2-2 \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = \vec{0}", font_size=38).set_z_index(6)
        v2_simp = MathTex(r"\begin{bmatrix} 1 & 1 \\ 0 & 0 \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix} \implies x + y = 0", font_size=36).set_z_index(6)
        v2_final = MathTex(r"\Rightarrow \vec{v}_2 = \begin{bmatrix} -1 \\ 1 \end{bmatrix}", font_size=42, color=BLUE_C).set_z_index(6)
        v2_final.next_to(v2_simp, DOWN, buff=0.5)

        self.play(Write(v2_head))
        self.play(FadeIn(v2_sys, shift=DOWN*0.2))
        self.wait(1)
        self.play(Transform(v2_sys, v2_simp))
        self.wait(1)
        self.play(Write(v2_final))
        self.play(Indicate(v2_final))
        self.wait(1.5)

        # Chuẩn bị chuyển sang hình học
        self.show_subtitle("Các thành phần đại số đã sẵn sàng. Hãy đưa chúng lên không gian.")
        
        # 3. CHUYỂN ĐỔI SANG HÌNH HỌC (Mapping)
        v1_vec = np.array([1, 0, 0])
        v2_vec = np.array([-1, 1, 0])

        kw_arrow = dict(buff=0, stroke_width=5, tip_length=0.25)
        self.vec_1 = Arrow(ORIGIN, v1_vec, color=RED_C, **kw_arrow).set_z_index(2)
        self.vec_2 = Arrow(ORIGIN, v2_vec, color=BLUE_C, **kw_arrow).set_z_index(2)
        self.lbl_v1 = MathTex(r"\vec{v}_1", font_size=32, color=RED_C).next_to(self.vec_1.get_end(), DR, buff=0.1).set_z_index(2)
        self.lbl_v2 = MathTex(r"\vec{v}_2", font_size=32, color=BLUE_C).next_to(self.vec_2.get_end(), UL, buff=0.1).set_z_index(2)

        # Hiệu ứng: Biến các con số ma trận thành mũi tên thực tế
        self.play(
            FadeOut(veil),
            FadeOut(lambda_storage),
            FadeOut(v2_head), FadeOut(v2_sys),
            ReplacementTransform(v1_storage, self.vec_1),
            ReplacementTransform(v2_final, self.vec_2),
            run_time=2
        )
        self.play(Write(self.lbl_v1), Write(self.lbl_v2))

        # 4. VẼ ĐƯỜNG ĐỨT NÉT VÀ KIỂM TRA TRỰC GIAO (Giữ lại cho bước sau)
        self.line_v1 = DashedLine(LEFT*10, RIGHT*10, color=RED_E, stroke_width=2, dash_length=0.1)
        self.line_v2 = DashedLine(v2_vec*10, -v2_vec*10, color=BLUE_E, stroke_width=2, dash_length=0.1)
        self.play(Create(self.line_v1), Create(self.line_v2), run_time=1.5)

        self.show_subtitle("Lưu ý: Hai vector này rõ ràng không vuông góc với nhau.")
        
        arc = Angle(self.vec_1, self.vec_2, radius=0.6, color=YELLOW, stroke_width=2.5)
        lbl_arc = MathTex(r"\neq 90^\circ", font_size=28, color=YELLOW).next_to(arc, UR, buff=0.1)
        
        self.play(Create(arc), Write(lbl_arc))
        self.wait(2)
        self.play(FadeOut(arc), FadeOut(lbl_arc))

    def beat_4_2_eigen_transformation(self):
        """Nhịp 4.2: Phép biến đổi Eigen"""
        self.show_subtitle("Sức mạnh của chúng nằm ở đâu? Hãy quan sát khi ma trận A làm thay đổi không gian.", run_time=1.5)

        self.lbl_v1.add_updater(lambda m: m.next_to(self.vec_1.get_end(), DR, buff=0.1))
        self.lbl_v2.add_updater(lambda m: m.next_to(self.vec_2.get_end(), UL, buff=0.1))

        self.play(
            ApplyMatrix(A_3x3, self.plane),
            ApplyMatrix(A_3x3, self.vec_1),
            ApplyMatrix(A_3x3, self.vec_2),
            run_time=4.0,
            rate_func=smooth
        )
        
        self.show_subtitle("Mọi thứ bị thay đổi, nhưng hai vector này vẫn nằm trên quỹ đạo ban đầu.", run_time=1.5)
        self.wait(1.5)

        lbl_lambda1 = create_text_with_bg("\u03bb\u2081 = 3", font_size=28, color=RED_B, weight=BOLD)
        lbl_lambda1.next_to(self.vec_1.get_end(), UP, buff=0.2)
        
        lbl_lambda2 = create_text_with_bg("\u03bb\u2082 = 2", font_size=28, color=BLUE_B, weight=BOLD)
        lbl_lambda2.next_to(self.vec_2.get_end(), RIGHT, buff=0.2)

        self.play(FadeIn(lbl_lambda1, shift=UP*0.1), FadeIn(lbl_lambda2, shift=RIGHT*0.1), run_time=1.0)
        self.show_subtitle("Chúng chỉ bị kéo giãn với tỷ lệ chính là các Giá trị riêng (Eigenvalues).", run_time=2.0)
        self.wait(2.0)

        self.lbl_v1.clear_updaters()
        self.lbl_v2.clear_updaters()
        
        self.play(
            FadeOut(self.plane), FadeOut(self.vec_1), FadeOut(self.vec_2),
            FadeOut(self.lbl_v1), FadeOut(self.lbl_v2),
            FadeOut(lbl_lambda1), FadeOut(lbl_lambda2),
            FadeOut(self.line_v1), FadeOut(self.line_v2),
            run_time=1.0
        )

    def beat_4_3_matrix_assembly(self):
        """NHỊP 4.3: TRÌNH BÀY PHÉP PHÂN TÍCH A = PDP^-1"""
        self.show_subtitle("Giờ ta lắp ráp chúng thành phương trình ma trận hoàn chỉnh.", run_time=1.5)

        mat_A = MathTex(r"\begin{bmatrix} 3 & 1 \\ 0 & 2 \end{bmatrix}", font_size=42, color=ORANGE)
        lbl_A = MathTex("A", font_size=32, color=ORANGE)
        grp_A = VGroup(lbl_A, mat_A).arrange(DOWN, buff=0.2)

        eq = MathTex("=", font_size=42, color=WHITE)

        mat_P = MathTex(r"\begin{bmatrix} 1 & -1 \\ 0 & 1 \end{bmatrix}", font_size=42, color=TEAL)
        lbl_P = MathTex("P", font_size=32, color=TEAL)
        grp_P = VGroup(lbl_P, mat_P).arrange(DOWN, buff=0.2)

        mat_D = MathTex(r"\begin{bmatrix} 3 & 0 \\ 0 & 2 \end{bmatrix}", font_size=42, color=YELLOW)
        lbl_D = MathTex("D", font_size=32, color=YELLOW)
        grp_D = VGroup(lbl_D, mat_D).arrange(DOWN, buff=0.2)

        mat_P_inv = MathTex(r"\begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}", font_size=42, color=TEAL_E)
        lbl_P_inv = MathTex("P^{-1}", font_size=32, color=TEAL_E)
        grp_P_inv = VGroup(lbl_P_inv, mat_P_inv).arrange(DOWN, buff=0.2)

        full_math_group = VGroup(grp_A, eq, grp_P, grp_D, grp_P_inv).arrange(RIGHT, buff=0.25)
        full_math_group.move_to(ORIGIN)

        self.play(FadeIn(grp_A), run_time=1.0)
        self.play(Write(eq), run_time=0.5)

        self.show_subtitle("Ma trận P được ghép từ các Vector riêng...", run_time=1.5)
        self.play(FadeIn(grp_P, shift=LEFT*0.2), run_time=1.0)
        self.wait(0.5)

        self.show_subtitle("...Ma trận đường chéo D chứa các Giá trị riêng...", run_time=1.5)
        self.play(FadeIn(grp_D, shift=LEFT*0.2), run_time=1.0)
        self.wait(0.5)

        self.show_subtitle("...và cuối cùng là nghịch đảo của P.", run_time=1.0)
        self.play(FadeIn(grp_P_inv, shift=LEFT*0.2), run_time=1.0)
        self.wait(2.0)

        self.show_subtitle("Phép phân tích Chéo hóa đã hoàn tất.", run_time=1.5)
        self.play(FadeOut(full_math_group), run_time=1.0)

    def beat_4_4_the_ultimate_comparison(self):
        """Nhịp 4.4: Bảng đối chiếu bản chất"""
        self.clear_subtitle()
        self.play(self.formula_grp.animate.to_edge(UP, buff=0.2).scale(0.8), run_time=1.0)

        title_criteria = create_text_with_bg("TIÊU CHÍ CỐT LÕI", font_size=24, color=YELLOW, weight=BOLD)
        title_svd = create_text_with_bg("SVD (A = U \u03a3 V\u1d40)", font_size=24, color=GREEN_B, weight=BOLD)
        title_diag = create_text_with_bg("CHÉO HÓA (A = P D P\u207b\u00b9)", font_size=24, color=BLUE_B, weight=BOLD)

        row0 = VGroup(title_criteria, title_svd, title_diag).arrange(RIGHT, buff=1.0)

        c1_1 = create_text_with_bg("Miền xác định", font_size=22, color=WHITE, weight=BOLD)
        c1_2 = create_text_with_bg("MỌI ma trận\n(Chữ nhật, suy biến)", font_size=20, color=WHITE)
        c1_3 = create_text_with_bg("CHỈ ma trận vuông\n(Đủ vector độc lập)", font_size=20, color=WHITE)
        
        c1_1.move_to(np.array([title_criteria.get_center()[0], -1, 0]))
        c1_2.move_to(np.array([title_svd.get_center()[0], -1, 0]))
        c1_3.move_to(np.array([title_diag.get_center()[0], -1, 0]))
        row1 = VGroup(c1_1, c1_2, c1_3)

        c2_1 = create_text_with_bg("Hệ cơ sở", font_size=22, color=WHITE, weight=BOLD)
        c2_2 = create_text_with_bg("Trực giao hoàn hảo\n(Luôn vuông góc)", font_size=20, color=GREEN_C)
        c2_3 = create_text_with_bg("Thường nghiêng ngả\n(Skewed basis)", font_size=20, color=RED_C)
        
        c2_1.move_to(np.array([title_criteria.get_center()[0], -2.5, 0]))
        c2_2.move_to(np.array([title_svd.get_center()[0], -2.5, 0]))
        c2_3.move_to(np.array([title_diag.get_center()[0], -2.5, 0]))
        row2 = VGroup(c2_1, c2_2, c2_3)

        table_grp = VGroup(row0, row1, row2).move_to(ORIGIN).shift(DOWN*0.5)

        line1 = Line(table_grp.get_left() + LEFT*0.5, table_grp.get_right() + RIGHT*0.5, color=GREY, stroke_width=1.5)
        line1.move_to(np.array([0, -0.1, 0]))
        line2 = Line(table_grp.get_left() + LEFT*0.5, table_grp.get_right() + RIGHT*0.5, color=GREY, stroke_width=1.5)
        line2.move_to(np.array([0, -1.75, 0]))

        self.play(FadeIn(row0, shift=DOWN*0.2), run_time=1.0)
        self.play(Create(line1), FadeIn(row1, shift=DOWN*0.2), run_time=1.2)
        
        self.show_subtitle("Tuy nhiên, Chéo hóa sụp đổ nếu ma trận không vuông hoặc thiếu vector độc lập.", run_time=2.0)
        self.wait(1.0)

        self.play(Create(line2), FadeIn(row2, shift=DOWN*0.2), run_time=1.2)
        
        self.show_subtitle("Ngược lại, SVD chấp nhận mọi ma trận và cung cấp một hệ cơ sở trực giao.", run_time=2.0)
        self.wait(1.5)

        box_svd_winner = SurroundingRectangle(c1_2, color=YELLOW, buff=0.1, stroke_width=3)
        box_svd_winner2 = SurroundingRectangle(c2_2, color=YELLOW, buff=0.1, stroke_width=3)
        
        self.play(Create(box_svd_winner), Create(box_svd_winner2), run_time=1.0)
        self.show_subtitle("Đó là lý do SVD trở thành nền tảng tối thượng của Tính toán khoa học.", run_time=2.5)
        
        self.wait(3.0)

        veil = FullScreenRectangle().set_fill(BLACK, opacity=1.0).set_z_index(10)
        self.play(FadeIn(veil), run_time=2.0)
        self.wait(1.0)