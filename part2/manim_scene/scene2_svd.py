from manim import *
import numpy as np

# Tính toán SVD chuẩn xác bằng Numpy
A_mat = np.array([[3.0, 1.0], 
                  [0.0, 2.0]])
U, Sigma, V_T = np.linalg.svd(A_mat)

# Bảng màu
C_I   = RED       
C_J   = GREEN_B   
C_MAT = ORANGE    

def create_text_with_bg(text, font_size=20, color=WHITE, bg_opacity=0.8, font="Cambria", weight=NORMAL):
    txt_obj = MarkupText(text, font=font, font_size=font_size, color=color, weight=weight).set_z_index(1)
    bg = BackgroundRectangle(txt_obj, color=BLACK, fill_opacity=bg_opacity, buff=0.15).set_z_index(0)
    return VGroup(bg, txt_obj)

class Scene2_SVD_Visualization(MovingCameraScene):
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
        self.setup_transition()
        self.beat_2_1_rotate_V_T()
        self.beat_2_2_scale_Sigma()
        self.beat_2_3_rotate_U()

    def setup_transition(self):
        """Bối cảnh chuyển giao: Intro Chapter 2 -> Lưới tọa độ chuẩn, công thức."""
        # 1. INTRO: Màn hình đen & Tiêu đề chương
        veil = FullScreenRectangle().set_fill(BLACK, opacity=1.0).set_z_index(10)
        self.add(veil) # Che toàn bộ màn hình khi mới vào Scene

        chapter_title = Text("Chapter 2", font_size=36, color=GREY_A).set_z_index(11)
        chapter_name = Text("The Tripartite Machine", font_size=48, color=WHITE, weight=BOLD).set_z_index(11)
        title_group = VGroup(chapter_title, chapter_name).arrange(DOWN, buff=0.3)
        
        self.play(FadeIn(title_group, shift=UP*0.2), run_time=1.5)
        self.wait(1.5)
        
        # Câu dẫn dắt (Subtitles) trước khi mở rèm
        self.show_subtitle("Vậy chính xác thì ba ma trận U, Σ, Vᵀ làm gì với không gian của chúng ta?")
        self.wait(2.0)
        
        # Mở rèm đen & ẩn tiêu đề
        self.play(
            FadeOut(title_group),
            FadeOut(veil),
            run_time=1.5
        )

        # 2. XÂY DỰNG KHÔNG GIAN (Lần lượt từ từ)
        self.plane = NumberPlane(
            x_range=[-7, 7, 1], y_range=[-4, 4, 1],
            background_line_style={"stroke_color": GREY, "stroke_width": 1, "stroke_opacity": 0.35},
            axis_config={"stroke_color": GREY_B, "stroke_width": 1.5},
        )
        self.unit_circle = Circle(radius=1.0, color=WHITE, stroke_width=2.5)
        
        kw_arrow = dict(buff=0, stroke_width=4.5, tip_length=0.22, max_tip_length_to_length_ratio=0.6)
        self.vec_i = Arrow(ORIGIN, RIGHT, color=C_I, **kw_arrow)
        self.vec_j = Arrow(ORIGIN, UP, color=C_J, **kw_arrow)
        
        self.lbl_i = MathTex(r"\hat{i}", font_size=30, color=C_I).next_to(self.vec_i.get_end(), DR, buff=0.04)
        self.lbl_j = MathTex(r"\hat{j}", font_size=30, color=C_J).next_to(self.vec_j.get_end(), UL, buff=0.04)

        # Gắn updater để nhãn luôn dính vào đầu mũi tên khi biến đổi
        self.lbl_i.add_updater(lambda m: m.next_to(self.vec_i.get_end(), DR, buff=0.04))
        self.lbl_j.add_updater(lambda m: m.next_to(self.vec_j.get_end(), UL, buff=0.04))

        # Gom nhóm toàn bộ không gian để tiện thao tác tổng thể
        self.space_group = VGroup(self.plane, self.unit_circle, self.vec_i, self.vec_j)

        # Trình diễn việc vẽ không gian thay vì hiện đột ngột
        self.play(Create(self.plane, lag_ratio=0.1), run_time=1.5)
        self.play(Create(self.unit_circle), GrowArrow(self.vec_i), GrowArrow(self.vec_j), run_time=1.0)
        self.play(FadeIn(self.lbl_i), FadeIn(self.lbl_j), run_time=0.5)

        # 3. HIỆN CÔNG THỨC Ở GÓC
        self.formula_svd = MathTex(r"A", r"=", r"U", r"\Sigma", r"V^T", font_size=42, color=WHITE)
        self.formula_svd[0].set_color(C_MAT) # A
        self.formula_svd[2].set_color(BLUE)  # U
        self.formula_svd[3].set_color(YELLOW)# Sigma
        self.formula_svd[4].set_color(GREEN) # V^T
        
        self.formula_bg = BackgroundRectangle(self.formula_svd, color=BLACK, fill_opacity=0.8, buff=0.15)
        self.formula_grp = VGroup(self.formula_bg, self.formula_svd).to_edge(UP, buff=0.4)
        
        self.show_subtitle("Hãy phân tích lại cỗ máy này, bắt đầu từ phải sang trái.")
        self.play(FadeIn(self.formula_grp, shift=DOWN*0.2), run_time=1.0)
        self.wait(1.0)


    def beat_2_1_rotate_V_T(self):
        """Nhịp 2.1: Ma trận V^T – Vô lăng của không gian nguồn"""
        part_U = self.formula_svd[2]
        part_Sigma = self.formula_svd[3]
        part_V_T = self.formula_svd[4]

        # Ánh sáng làm mờ U và Sigma, bật V^T
        self.play(
            part_U.animate.set_opacity(0.3),
            part_Sigma.animate.set_opacity(0.3),
            Indicate(part_V_T, color=GREEN_B, scale_factor=1.3),
            run_time=1.0
        )
        
        self.show_subtitle("Bước một, thao tác trên không gian nguồn với ma trận V chuyển vị.", run_time=1.0)
        
        # Áp dụng V^T cơ bản của bài toán để đưa không gian vào vị trí
        matrix_V_T = np.array([
            [V_T[0, 0], V_T[0, 1], 0],
            [V_T[1, 0], V_T[1, 1], 0],
            [0,        0,        1]
        ])
        self.play(ApplyMatrix(matrix_V_T, self.space_group), run_time=2.0)
        self.wait(0.5)

        # -----------------------------------------------------------------
        # PHẪU THUẬT THAM SỐ: Hiện Ma trận Động và quay thật chậm
        # -----------------------------------------------------------------
        self.show_subtitle("Thử thay đổi góc xoay tham số của nó. Chú ý: Hình tròn vẫn là hình tròn.", run_time=1.5)
        
        theta_v = ValueTracker(0)
        
        # Các con số trong ma trận (dùng DecimalNumber để tự nhảy số)
        v00 = DecimalNumber(1, num_decimal_places=2, font_size=40).set_color(GREEN_B)
        v01 = DecimalNumber(0, num_decimal_places=2, font_size=40).set_color(GREEN_B)
        v10 = DecimalNumber(0, num_decimal_places=2, font_size=40).set_color(GREEN_B)
        v11 = DecimalNumber(1, num_decimal_places=2, font_size=40).set_color(GREEN_B)

        v00.add_updater(lambda d: d.set_value(np.cos(theta_v.get_value())))
        v01.add_updater(lambda d: d.set_value(-np.sin(theta_v.get_value())))
        v10.add_updater(lambda d: d.set_value(np.sin(theta_v.get_value())))
        v11.add_updater(lambda d: d.set_value(np.cos(theta_v.get_value())))

        # Đóng gói ma trận
        vt_label = MathTex(r"V^T(\theta) = ", font_size=36, color=WHITE)
        row1 = VGroup(v00, v01).arrange(RIGHT, buff=0.8)
        row2 = VGroup(v10, v11).arrange(RIGHT, buff=0.8)
        matrix_elements = VGroup(row1, row2).arrange(DOWN, buff=0.4)
        
        left_bracket = MathTex("[", font_size=72, color=WHITE).next_to(matrix_elements, LEFT, buff=0.1)
        right_bracket = MathTex("]", font_size=72, color=WHITE).next_to(matrix_elements, RIGHT, buff=0.1)
        vt_label.next_to(left_bracket, LEFT, buff=0.2)

        dyn_matrix_grp = VGroup(vt_label, left_bracket, matrix_elements, right_bracket)
        dyn_matrix_grp.to_corner(UL, buff=0.5).shift(DOWN*0.5)
        bg_matrix = BackgroundRectangle(dyn_matrix_grp, color=BLACK, fill_opacity=0.85, buff=0.25)
        panel_V = VGroup(bg_matrix, dyn_matrix_grp)
        
        self.play(FadeIn(panel_V, shift=RIGHT*0.2), run_time=1.0)

        # Vẽ một góc vuông màu vàng bám chặt vào hệ cơ sở
        right_angle = always_redraw(lambda: RightAngle(self.vec_i, self.vec_j, length=0.25, color=YELLOW, stroke_width=3))
        self.play(Create(right_angle))

        # QUAY KHÔNG GIAN 1 VÒNG VÀ CHO SỐ NHẢY ĐỒNG THỜI (10 GIÂY)
        self.play(
            theta_v.animate.set_value(2*PI),
            Rotate(self.space_group, angle=2*PI, about_point=ORIGIN), 
            run_time=10.0, 
            rate_func=linear
        )
        
        self.show_subtitle("Nhiệm vụ duy nhất của nó là dò tìm và định vị hệ cơ sở, chuẩn bị cho phép biến đổi.", run_time=1.0)
        self.wait(2.0)
        
        # Dọn dẹp để sang bước tiếp theo
        self.play(FadeOut(right_angle), FadeOut(panel_V), run_time=1.0)


    def beat_2_2_scale_Sigma(self):
        """Nhịp 2.2: Ma trận Sigma – Nhịp đập và sự sụp đổ chiều không gian"""
        part_V_T = self.formula_svd[4]
        part_Sigma = self.formula_svd[3]

        self.play(
            part_V_T.animate.set_opacity(0.3),
            part_Sigma.animate.set_opacity(1.0),
            Indicate(part_Sigma, color=YELLOW, scale_factor=1.3),
            run_time=1.0
        )

        self.show_subtitle("Bước hai: Ma trận đường chéo Sigma - sự co giãn.", run_time=1.0)

        # -----------------------------------------------------------------
        # PHẪU THUẬT THAM SỐ: Hiện Ma trận Động của Sigma
        # -----------------------------------------------------------------
        sig1_tracker = ValueTracker(1.0) # Khởi đầu hình tròn có bán kính 1
        sig2_tracker = ValueTracker(1.0)

        s11 = DecimalNumber(1, num_decimal_places=2, font_size=40).set_color(YELLOW)
        s12 = MathTex("0", font_size=40, color=WHITE)
        s21 = MathTex("0", font_size=40, color=WHITE)
        s22 = DecimalNumber(1, num_decimal_places=2, font_size=40).set_color(YELLOW)

        s11.add_updater(lambda d: d.set_value(sig1_tracker.get_value()))
        s22.add_updater(lambda d: d.set_value(sig2_tracker.get_value()))

        sig_label = MathTex(r"\Sigma = ", font_size=36, color=WHITE)
        row1 = VGroup(s11, s12).arrange(RIGHT, buff=0.8)
        row2 = VGroup(s21, s22).arrange(RIGHT, buff=0.8)
        matrix_elements = VGroup(row1, row2).arrange(DOWN, buff=0.4)
        
        left_bracket = MathTex("[", font_size=72, color=WHITE).next_to(matrix_elements, LEFT, buff=0.1)
        right_bracket = MathTex("]", font_size=72, color=WHITE).next_to(matrix_elements, RIGHT, buff=0.1)
        sig_label.next_to(left_bracket, LEFT, buff=0.2)

        dyn_matrix_grp = VGroup(sig_label, left_bracket, matrix_elements, right_bracket)
        dyn_matrix_grp.to_corner(UL, buff=0.5).shift(DOWN*0.5)
        bg_matrix = BackgroundRectangle(dyn_matrix_grp, color=BLACK, fill_opacity=0.85, buff=0.25)
        panel_S = VGroup(bg_matrix, dyn_matrix_grp)
        
        self.play(FadeIn(panel_S, shift=RIGHT*0.2), run_time=1.0)

        # 1. Kéo hình tròn thành Elip của bài toán và cập nhật số
        self.play(
            sig1_tracker.animate.set_value(Sigma[0]),
            sig2_tracker.animate.set_value(Sigma[1]),
            self.space_group.animate.stretch(Sigma[0], dim=0).stretch(Sigma[1], dim=1),
            self.unit_circle.animate.set_color(YELLOW),
            run_time=3.0,
            rate_func=smooth
        )
        self.wait(1.0)

        # 2. Hít thở
        self.show_subtitle("Khi thay đổi các số suy biến, không gian thay đổi dọc theo trục hoành...", run_time=1.0)
        self.play(
            sig1_tracker.animate.set_value(Sigma[0] * 1.3),
            self.space_group.animate.stretch(1.3, dim=0), 
            run_time=1.5
        )
        self.play(
            sig1_tracker.animate.set_value(Sigma[0]),
            self.space_group.animate.stretch(1/1.3, dim=0), 
            run_time=1.5
        )
        
        self.show_subtitle("...và thay đổi dọc theo trục tung.", run_time=1.0)
        self.play(
            sig2_tracker.animate.set_value(Sigma[1] * 1.3),
            self.space_group.animate.stretch(1.3, dim=1), 
            run_time=1.5
        )
        self.play(
            sig2_tracker.animate.set_value(Sigma[1]),
            self.space_group.animate.stretch(1/1.3, dim=1), 
            run_time=1.5
        )
        self.wait(0.5)

        # 3. Sự sụp đổ (Collapse to 1D)
        self.space_group.save_state()
        self.show_subtitle("Nhưng điều gì xảy ra nếu ta ép sigma 2 về 0?", run_time=1.0)
        
        warning_text = create_text_with_bg("Cảnh báo: Nếu \u03c3\u2082 = 0 \u2192 Suy biến 1D", font_size=24, color=RED_B, weight=BOLD)
        warning_text.to_corner(DR, buff=0.5)
        self.play(FadeIn(warning_text, shift=UP*0.2))

        # Ép dẹt trục Y và ép số về 0
        self.play(
            sig2_tracker.animate.set_value(0.0),
            self.space_group.animate.stretch(0.001, dim=1), 
            run_time=4.0
        )
        
        self.show_subtitle("Không gian bị biến đổi thành một đường thẳng. Chiều thứ hai đã mất.", run_time=1.0)
        self.wait(3.0)

        self.show_subtitle("Giờ ta trả không gian về lại giá trị Sigma thật của bài toán.", run_time=1.0)
        self.play(
            sig2_tracker.animate.set_value(Sigma[1]),
            Restore(self.space_group), 
            FadeOut(warning_text), 
            run_time=2.5
        )
        self.wait(1.0)
        self.play(FadeOut(panel_S))


    def beat_2_3_rotate_U(self):
        """Nhịp 2.3: Ma trận U – Cánh quạt định vị đích"""
        part_Sigma = self.formula_svd[3]
        part_U = self.formula_svd[2]

        self.play(
            part_Sigma.animate.set_opacity(0.3),
            part_U.animate.set_opacity(1.0),
            Indicate(part_U, color=BLUE, scale_factor=1.3),
            run_time=1.0
        )

        self.show_subtitle("Cuối cùng là U - một ma trận trực giao khác tiếp quản.", run_time=1.0)

        # -----------------------------------------------------------------
        # PHẪU THUẬT THAM SỐ: Hiện Ma trận Động của U
        # -----------------------------------------------------------------
        theta_u = ValueTracker(0)
        
        u00 = DecimalNumber(1, num_decimal_places=2, font_size=40).set_color(BLUE)
        u01 = DecimalNumber(0, num_decimal_places=2, font_size=40).set_color(BLUE)
        u10 = DecimalNumber(0, num_decimal_places=2, font_size=40).set_color(BLUE)
        u11 = DecimalNumber(1, num_decimal_places=2, font_size=40).set_color(BLUE)

        u00.add_updater(lambda d: d.set_value(np.cos(theta_u.get_value())))
        u01.add_updater(lambda d: d.set_value(-np.sin(theta_u.get_value())))
        u10.add_updater(lambda d: d.set_value(np.sin(theta_u.get_value())))
        u11.add_updater(lambda d: d.set_value(np.cos(theta_u.get_value())))

        u_label = MathTex(r"U(\theta) = ", font_size=36, color=WHITE)
        row1 = VGroup(u00, u01).arrange(RIGHT, buff=0.8)
        row2 = VGroup(u10, u11).arrange(RIGHT, buff=0.8)
        matrix_elements = VGroup(row1, row2).arrange(DOWN, buff=0.4)
        
        left_bracket = MathTex("[", font_size=72, color=WHITE).next_to(matrix_elements, LEFT, buff=0.1)
        right_bracket = MathTex("]", font_size=72, color=WHITE).next_to(matrix_elements, RIGHT, buff=0.1)
        u_label.next_to(left_bracket, LEFT, buff=0.2)

        dyn_matrix_grp = VGroup(u_label, left_bracket, matrix_elements, right_bracket)
        dyn_matrix_grp.to_corner(UL, buff=0.5).shift(DOWN*0.5)
        bg_matrix = BackgroundRectangle(dyn_matrix_grp, color=BLACK, fill_opacity=0.85, buff=0.25)
        panel_U = VGroup(bg_matrix, dyn_matrix_grp)
        
        self.play(FadeIn(panel_U, shift=RIGHT*0.2), run_time=1.0)

        self.show_subtitle("Tương tự, U xoay khối elip này thực hiện phép quay.", run_time=1.5)
        
        # Quay khối Elip như cánh quạt (lắc qua lắc lại chậm rãi)
        self.play(
            theta_u.animate.set_value(PI/4),
            Rotate(self.space_group, angle=PI/4, about_point=ORIGIN), 
            run_time=3.0, rate_func=smooth
        )
        self.play(
            theta_u.animate.set_value(-PI/6),
            Rotate(self.space_group, angle=-PI/4 - PI/6, about_point=ORIGIN), 
            run_time=4.0, rate_func=smooth
        )
        
        # Xoay về vị trí đích thực tế của ma trận A
        # Tính góc của ma trận U thực tế để khớp hình
        actual_angle = np.arctan2(U[1,0], U[0,0])
        self.play(
            theta_u.animate.set_value(actual_angle),
            Rotate(self.space_group, angle=actual_angle - (-PI/6), about_point=ORIGIN), 
            run_time=3.0, rate_func=smooth
        )

        # 3. Kết luận toàn cảnh
        self.play(FadeOut(panel_U))
        self.clear_subtitle()
        
        self.play(
            self.formula_svd[0].animate.set_opacity(1.0),
            self.formula_svd[3].animate.set_opacity(1.0),
            self.formula_svd[4].animate.set_opacity(1.0),
            run_time=1.0
        )

        self.show_subtitle("Xoay. Kéo giãn. Và xoay.", run_time=1.0)
        self.play(Indicate(self.formula_svd[4], color=GREEN_B), run_time=0.6)
        self.play(Indicate(self.formula_svd[3], color=YELLOW), run_time=0.6)
        self.play(Indicate(self.formula_svd[2], color=BLUE), run_time=0.6)
        
        self.show_subtitle("Phép màu của SVD đã giải mã phép biến đồi A hình dung thành 3 bước đơn giản.", run_time=1.0)
        self.wait(3.0)

        self.lbl_i.clear_updaters()
        self.lbl_j.clear_updaters()
        veil = FullScreenRectangle().set_fill(BLACK, opacity=1.0).set_z_index(10)
        self.play(FadeIn(veil), run_time=1.5)
        self.wait(1.0)