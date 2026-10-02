# Weights Directory

This directory stores model checkpoint files.

## Contents

| File | Model | Size | Source |
|:---|:---|:---|:---|
| `yolov8n.pt` | YOLOv8 Nano (COCO pre-trained) | ~6.2 MB | [Ultralytics](https://github.com/ultralytics/ultralytics) |

## Notes

- **YOLOv8n** is the nano variant (3.15M parameters, 8.7 GFLOPs) — the smallest and fastest.
- Weights are auto-downloaded by Ultralytics if not present. You can also download manually:
  ```python
  from ultralytics import YOLO
  model = YOLO("yolov8n.pt")  # Auto-downloads to current directory
  ```
- **Depth Anything V2** weights are downloaded automatically from HuggingFace at runtime.
  The model used is `depth-anything/Depth-Anything-V2-Small-hf` (24.8M parameters).
  Cached in `~/.cache/huggingface/hub/`.
