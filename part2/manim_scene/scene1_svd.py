from manim import *
import numpy as np

# 1. Tính toán SVD chuẩn xác bằng Numpy để vẽ trục Elip
A_mat = np.array([[3.0, 1.0], 
                  [0.0, 2.0]])
U, Sigma, V_T = np.linalg.svd(A_mat)

# Tính tọa độ đích của các trục elip (sigma_i * u_i)
AXIS_1_END = U[:, 0] * Sigma[0]
AXIS_2_END = U[:, 1] * Sigma[1]

# Tính góc thực tế giữa Ai và Aj
Ai = A_mat @ np.array([1, 0])
Aj = A_mat @ np.array([0, 1])
cos_theta = np.dot(Ai, Aj) / (np.linalg.norm(Ai) * np.linalg.norm(Aj))
ANGLE_DEG_STR = f"{np.degrees(np.arccos(cos_theta)):.1f}^\circ"

# Ma trận 3D cho Manim
A_3x3 = np.array([[3, 1, 0],
                  [0, 2, 0],
                  [0, 0, 1]], dtype=float)

# Bảng màu
C_I   = RED       
C_J   = GREEN_B   
C_ELP = YELLOW    
C_MAT = ORANGE    

def create_text_with_bg(text, font_size=20, color=WHITE, bg_opacity=0.8, font="Cambria", weight=NORMAL):
    txt_obj = MarkupText(text, font=font, font_size=font_size, color=color, weight=weight).set_z_index(1)
    
    # Lấy khung bao chuẩn, mở rộng ra một chút
    bg = BackgroundRectangle(txt_obj, color=BLACK, fill_opacity=bg_opacity, buff=0.15).set_z_index(0)
    return VGroup(bg, txt_obj)


class Scene1_AnatomyOfChaos(MovingCameraScene):
    def show_subtitle(self, text, run_time=1.0):
        """Hàm hỗ trợ hiện phụ đề ở cạnh dưới màn hình"""
        subtitle = create_text_with_bg(text, font_size=18, color=WHITE, font="Cambria")
        
        # Đẩy xuống dưới cùng
        subtitle.to_edge(DOWN, buff=0.4) 
        
        # Nếu có phụ đề cũ đang hiện, thay thế nó
        if hasattr(self, "current_subtitle"):
            self.play(Transform(self.current_subtitle, subtitle), run_time=run_time)
        else:
            self.play(FadeIn(subtitle, shift=UP*0.2), run_time=run_time)
            self.current_subtitle = subtitle

    def clear_subtitle(self, run_time=0.5):
        """Hàm xóa phụ đề khỏi màn hình"""
        if hasattr(self, "current_subtitle"):
            self.play(FadeOut(self.current_subtitle, shift=DOWN*0.2), run_time=run_time)
            del self.current_subtitle

    def construct(self):
        self.beat_0_intro()
        self.beat_1_1_orthonormal_basis()
        self.beat_1_2_linear_transformation()
        self.beat_1_3_loss_of_orthogonality()
        self.beat_1_4_core_question()

    # ── Phân cảnh 0: Giới thiệu chủ đề ──
    def beat_0_intro(self):
        # Tiêu đề chính
        title = Text("Singular Value Decomposition",
                      font_size=48, color=WHITE, weight=BOLD)
        # Viết tắt (SVD)
        abbr = Text("(SVD)", font_size=40, color=GOLD, weight=BOLD)
        abbr.next_to(title, DOWN, buff=0.25)

        # Đường kẻ trang trí
        line = Line(LEFT * 4, RIGHT * 4, color=GOLD, stroke_width=2)
        line.next_to(abbr, DOWN, buff=0.35)

        # Phụ đề
        subtitle = Text("Chapter 1 — Anatomy of Chaos",
                         font_size=28, color=GREY_A)
        subtitle.next_to(line, DOWN, buff=0.35)

        # Công thức SVD nhỏ bên dưới
        formula = MathTex(r"A = U \, \Sigma \, V^T", font_size=36, color=BLUE_B)
        formula.next_to(subtitle, DOWN, buff=0.5)

        # ── Animation ──
        self.play(Write(title), run_time=1.5)
        self.play(FadeIn(abbr, shift=UP * 0.15), run_time=0.9)
        self.play(Create(line), run_time=0.6)
        self.play(FadeIn(subtitle, shift=UP * 0.15), run_time=0.9)
        self.play(Write(formula), run_time=1.2)
        self.wait(2)

        # Fade out toàn bộ
        intro_group = VGroup(title, abbr, line, subtitle, formula)
        self.play(FadeOut(intro_group, shift=UP * 0.3), run_time=1.2)
        self.wait(0.5)

    def beat_1_1_orthonormal_basis(self):
        self.show_subtitle("Để giải mã một phép biến đổi phức tạp, chúng ta phải bắt đầu từ những thứ cơ bản nhất.")
        # Tạo hệ tọa độ Descartes
        plane = NumberPlane(
            # Phạm vi hiển thị (từ -7 đến 7 trên trục x, từ -4 đến 4 trên trục y)
            x_range=[-7, 7, 1], y_range=[-4, 4, 1],
            # Đường lưới
            background_line_style={"stroke_color": GREY, "stroke_width": 1, "stroke_opacity": 0.35},
            # Trục tọa độ (màu xám đậm, độ dày 1.5)
            axis_config={"stroke_color": GREY_B, "stroke_width": 1.5},
        )
        self.show_subtitle("Hãy quan sát không gian hai chiều này.")
        # Vẽ hệ tọa độ
        self.play(Create(plane, lag_ratio=0.05), run_time=2.25)
        self.wait(0.5)
        self.show_subtitle("Bắt đầu từ đường tròn đơn vị, nơi mọi vector đều có chiều dài chuẩn xác bằng 1")
        
        # Vẽ đường tròn đơn vị (tâm tại gốc tọa độ, bán kính 1)
        unit_circle = Circle(radius=1.0, color=WHITE, stroke_width=2.5)
        lbl_circle = MathTex(r"\|x\|_2 = 1", font_size=26, color=WHITE).move_to([1.55, 0.9, 0])
        self.play(Create(unit_circle), Write(lbl_circle), run_time=1.5)

        self.wait(0.5)
        self.show_subtitle("Tại lõi của nó, hai vector cơ sở i và j tạo thành một góc vông")
        

        # Vẽ vector i và j (độ dài 1, hướng sang phải và lên trên)
        kw_arrow = dict(buff=0, stroke_width=4.5, tip_length=0.22, max_tip_length_to_length_ratio=0.6)
        vec_i = Arrow(ORIGIN, RIGHT, color=C_I, **kw_arrow)
        vec_j = Arrow(ORIGIN, UP, color=C_J, **kw_arrow)
        lbl_i = MathTex(r"\hat{i}", font_size=30, color=C_I).next_to(RIGHT * 1.05, DR, buff=0.04)
        lbl_j = MathTex(r"\hat{j}", font_size=30, color=C_J).next_to(UP   * 1.05, UL, buff=0.04)

        # Vẽ vector i và j
        self.play(GrowArrow(vec_i), GrowArrow(vec_j), run_time=1.5)
        self.play(Write(lbl_i), Write(lbl_j), run_time=0.9)

        # Vẽ góc vuông giữa vector i và j
        right_angle = RightAngle(vec_i, vec_j, length=0.18, color=YELLOW, stroke_width=2)
        lbl_90 = MathTex(r"90^\circ", font_size=22, color=YELLOW).shift(RIGHT * 0.32 + UP * 0.32)
        self.play(Create(right_angle), Write(lbl_90), run_time=1.05)
        self.wait(1.5)
        
        self.show_subtitle("Đây là nền tảng của không gian")

        # Lưu lại các đối tượng để sử dụng trong các phân cảnh sau
        self.plane, self.unit_circle, self.lbl_circle = plane, unit_circle, lbl_circle
        self.vec_i, self.lbl_i, self.vec_j, self.lbl_j = vec_i, lbl_i, vec_j, lbl_j
        self.right_angle, self.lbl_90 = right_angle, lbl_90

    # Phân cảnh 1.2: Biến đổi tuyến tính
    def beat_1_2_linear_transformation(self):
        # Tạo ma trận A
        mat_tex = MathTex(r"A = \begin{bmatrix} 3 & 1 \\ 0 & 2 \end{bmatrix}", font_size=38, color=C_MAT)
        # Đặt ma trận A ở góc trên bên trái
        mat_tex.to_corner(UL, buff=0.4).add_background_rectangle(color=BLACK, opacity=0.8, buff=0.12)
        self.play(FadeIn(mat_tex[0]), Write(mat_tex[1]), run_time=1.5)
        self.show_subtitle("Xét ma trận A = [[3,1],[0,2]]. Đây là một phép biến đổi tuyến tính đơn giản.")
        self.wait(0.75)

        # Xóa góc vuông và nhãn góc vuông
        self.play(FadeOut(self.right_angle), FadeOut(self.lbl_90), FadeOut(self.lbl_circle), run_time=0.75)

        # Tạo bóng mờ của hệ quy chiếu ban đầu
        ghost_i = self.vec_i.copy().set_opacity(0.2).set_color(RED_E)
        ghost_j = self.vec_j.copy().set_opacity(0.2).set_color(GREEN_E)
        ghost_c = self.unit_circle.copy().set_opacity(0.15)
        self.ghosts = VGroup(ghost_i, ghost_j, ghost_c)
        self.add(self.ghosts)

        # Vẽ vector Ai và Aj
        kw = dict(buff=0, stroke_width=4.5, tip_length=0.22)
        # Vector Ai (độ dài 3, hướng sang phải)
        vec_Ai = Arrow(ORIGIN, RIGHT * 3, color=C_I, **kw)
        # Vector Aj (độ dài sqrt(5), hướng sang phải và lên trên)
        vec_Aj = Arrow(ORIGIN, RIGHT * 1 + UP * 2, color=C_J, **kw)
        # Nhãn vector Ai
        lbl_Ai = MathTex(r"A\hat{i} = [3,0]^T", font_size=24, color=C_I).next_to(RIGHT * 3.1, DR, buff=0.05)
        # Nhãn vector Aj
        lbl_Aj = MathTex(r"A\hat{j} = [1,2]^T", font_size=24, color=C_J).next_to(RIGHT * 1 + UP * 2.1, UL, buff=0.05)

        # Vẽ vector Ai và Aj
        self.play(
            # Biến đổi vector i và j thành vector Ai và Aj
            Transform(self.vec_i, vec_Ai), Transform(self.lbl_i, lbl_Ai),
            Transform(self.vec_j, vec_Aj), Transform(self.lbl_j, lbl_Aj),
            # Biến đổi đường tròn đơn vị thành elip ảnh của A
            ApplyMatrix(A_3x3, self.unit_circle),
            run_time=5.25, rate_func=smooth,
        )
        self.show_subtitle("A kéo giãn và xoay không gian — đường tròn đơn vị bị biến dạng thành một hình elip.")
        # Tô màu elip ảnh của A
        self.play(self.unit_circle.animate.set_color(C_ELP), run_time=0.6)

        # Tạo nhãn elip ảnh của A
        lbl_elp_grp = create_text_with_bg("Image of unit circle under A", font_size=22, color=C_ELP)
        lbl_elp_grp.move_to([2.3, -2.6, 0])
        self.play(FadeIn(lbl_elp_grp, shift=UP * 0.15), run_time=1.05)
        self.show_subtitle("Hình elip này chứa đựng toàn bộ nội dung hình học của phép biến đổi A.")
        self.wait(1.5)

        # Lưu lại các đối tượng để sử dụng trong các phân cảnh sau
        self.mat_tex, self.lbl_elp_grp = mat_tex, lbl_elp_grp

    # Phân cảnh 1.3: Mất tính trực giao
    def beat_1_3_loss_of_orthogonality(self):
        # Xóa bỏ tập hợp bóng mờ
        self.play(self.plane.animate.set_opacity(0.12), self.mat_tex.animate.set_opacity(0.3),
                  FadeOut(self.lbl_elp_grp), FadeOut(self.ghosts), run_time=1.35)

        # Vẽ góc lệch theta
        self.show_subtitle("Nhưng có điều gì đó đã bị phá vỡ. Góc vuông ban đầu giữa î và ĵ không còn nữa.")
        arc = Angle(self.vec_i, self.vec_j, radius=0.55, color=YELLOW_B, stroke_width=2.5)
        lbl_arc = MathTex(rf"\theta \approx {ANGLE_DEG_STR}", font_size=26, color=YELLOW_B)
        lbl_arc.move_to([0.85, 0.45, 0]).add_background_rectangle(color=BLACK, opacity=0.7, buff=0.08)
        self.play(Create(arc), Write(lbl_arc), run_time=1.5)

        # Vẽ biểu thức tích vô hướng
        dot_eq = MathTex(
            r"\langle A\hat{i},\, A\hat{j} \rangle", r"= 3 \cdot 1 + 0 \cdot 2", r"= 3", r"\neq 0",
            font_size=34
        )
        # Tô màu các thành phần của biểu thức
        dot_eq[0].set_color(WHITE); dot_eq[1].set_color(GREY_A)
        dot_eq[2].set_color(ORANGE); dot_eq[3].set_color(RED)
        # Sắp xếp các thành phần của biểu thức
        dot_eq.arrange(RIGHT, buff=0.22).to_edge(DOWN, buff=1.2)
        # Thêm nền cho biểu thức
        dot_eq.add_background_rectangle(color=BLACK, opacity=0.82, buff=0.16)

        # Vẽ khung bao quanh biểu thức
        neq_box = SurroundingRectangle(dot_eq[3], color=RED, buff=0.1, stroke_width=2, corner_radius=0.05)
        # Vẽ biểu thức và khung bao quanh
        self.play(Write(dot_eq), run_time=2.1)
        self.show_subtitle("Tích vô hướng ⟨Aî, Aĵ⟩ = 3 ≠ 0. Hai vector ảnh không còn trực giao với nhau.")

        self.play(Create(neq_box), Indicate(dot_eq[3], color=RED, scale_factor=1.25), run_time=1.2)
        self.wait(3.0)

        # Lưu lại các đối tượng để sử dụng trong các phân cảnh sau
        self.arc_13, self.lbl_arc, self.dot_eq, self.neq_box = arc, lbl_arc, dot_eq, neq_box

    # Phân cảnh 1.4: Câu hỏi cốt lõi
    def beat_1_4_core_question(self):
        # Phóng to camera
        self.play(self.camera.frame.animate.scale(1.3), run_time=1.5)
        # Xóa bỏ các đối tượng không cần thiết
        self.play(
            FadeOut(self.arc_13), FadeOut(self.lbl_arc), FadeOut(self.dot_eq), FadeOut(self.neq_box),
            self.vec_i.animate.set_opacity(0.35), self.vec_j.animate.set_opacity(0.35),
            self.lbl_i.animate.set_opacity(0.35), self.lbl_j.animate.set_opacity(0.35),
            run_time=1.35
        )

        # Tạo trục chính của elip
        ax1_vec = np.array([AXIS_1_END[0], AXIS_1_END[1], 0])
        ax2_vec = np.array([AXIS_2_END[0], AXIS_2_END[1], 0])
        self.show_subtitle("Tuy nhiên, elip luôn có các trục chính — và các trục này lại vuông góc với nhau!")

        # Vẽ trục chính của elip    
        ax1 = DashedLine(-ax1_vec, ax1_vec, color=BLUE_B, stroke_width=2.5, dash_length=0.13)
        ax2 = DashedLine(-ax2_vec, ax2_vec, color=TEAL_B, stroke_width=2.5, dash_length=0.13)

        # Vẽ nhãn trục chính của elip
        lbl_s1 = MathTex(r"\sigma_1 \vec{u}_1", font_size=26, color=BLUE_B).next_to(ax1_vec, RIGHT, buff=0.1)
        lbl_s2 = MathTex(r"\sigma_2 \vec{u}_2", font_size=26, color=TEAL_B).next_to(ax2_vec, UP, buff=0.1)

        # Vẽ nhãn trục chính của elip
        note_axes_grp = create_text_with_bg("Axes of the ellipse", font_size=20, color=WHITE)
        note_axes_grp.to_corner(UR, buff=0.35)

        # Vẽ trục chính của elip
        self.play(Create(ax1), Create(ax2), Write(lbl_s1), Write(lbl_s2), FadeIn(note_axes_grp, shift=LEFT * 0.15), run_time=2.1)
        self.wait(1.5)

        # Giữ lại trục và hình trung tâm, chỉ ẩn đi các nhãn phụ để tránh rối mắt
        to_fade = Group(
            self.mat_tex, self.lbl_i, self.lbl_j, note_axes_grp
        )
        self.play(FadeOut(to_fade), run_time=1.0)
        self.wait(0.5)

        # Vẽ câu hỏi cốt lõi
        question_grp = create_text_with_bg("How to recover orthogonality?", font_size=30, color=WHITE, bg_opacity=0.9)
        question_grp.to_edge(UP, buff=0)

        # Vẽ công thức SVD
        svd_key = MathTex(r"\text{Singular Value Decomposition (SVD):}\quad A = U\,\Sigma\,V^T", font_size=25, color=GOLD)
        svd_key.next_to(question_grp, DOWN, buff=0.15).add_background_rectangle(color=BLACK, opacity=0.88, buff=0.15)
        svd_uline = Underline(svd_key[1], color=GOLD, stroke_width=1.5)

        self.play(FadeIn(question_grp, shift=DOWN * 0.25), run_time=1.35)
        self.show_subtitle("Câu hỏi đặt ra: liệu có tồn tại một hệ cơ sở đầu vào mà A biến đổi thành một hệ trực giao?")
        self.play(Write(svd_key), Create(svd_uline), run_time=1.95)
        self.show_subtitle("SVD chính là câu trả lời: A = UΣVᵀ — phân tách phép biến đổi thành xoay, giãn, rồi xoay lại.")

        # Thêm hiệu ứng nhấn mạnh tính trực giao (orthogonality đã được phục hồi nhờ SVD)
        ax1_line_half = Line(ORIGIN, ax1_vec)
        ax2_line_half = Line(ORIGIN, ax2_vec)
        ax_right_angle = RightAngle(ax1_line_half, ax2_line_half, length=0.35, color=RED, stroke_width=3)
        lbl_ortho = MathTex(r"90^\circ", font_size=24, color=RED).next_to(ax_right_angle, UR, buff=0.08)

        self.play(
            Create(ax_right_angle), Write(lbl_ortho), 
            Indicate(ax1, color=GOLD, scale_factor=1.1),
            Indicate(ax2, color=GOLD, scale_factor=1.1),
            run_time=2.0
        )
        self.play(Circumscribe(svd_key[1], color=GOLD, fade_out=True), run_time=1.5)

        self.wait(1.75)

        # Ẩn toàn bộ các đối tượng còn lại trên màn hình
        to_fade = Group(
            question_grp, svd_key, svd_uline, ax_right_angle, lbl_ortho, 
            ax1, ax2, lbl_s1, lbl_s2, self.unit_circle, self.vec_i, self.vec_j, self.plane
        )
        self.play(FadeOut(to_fade), run_time=1.5)
        self.wait(0.5)