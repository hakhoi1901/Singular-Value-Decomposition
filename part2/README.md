# Manim

## Render video

```bash
# Render video
manim -pqh manim_scene.py Scene1_AnatomyOfChaos

# Render video
manim -pqh manim_scene.py Scene2_GeometricInterpretation

# Render video
manim -pqh manim_scene.py Scene3_SingularValues

# Render video
manim -pqh manim_scene.py Scene4_Applications
```

## Flags 

| Flag | Ý nghĩa |
|------|---------|
| `-p`  | Tự mở video sau khi render |
| `-ql` | Low quality (480p15) – nhanh |
| `-qm` | Medium quality (720p30) |
| `-qh` | High quality (1080p60) |
| `-s`  | Chỉ xuất ảnh frame cuối |
| `--format=gif` | Xuất GIF |
