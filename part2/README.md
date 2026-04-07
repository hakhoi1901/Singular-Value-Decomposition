# Manim

## Render video

```bash
# Chất lượng thấp (preview nhanh)
manim -pql manim_scene.py Scene1_AnatomyOfChaos

# Chất lượng cao (1080p)
manim -pqh manim_scene.py Scene1_AnatomyOfChaos
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
