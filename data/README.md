# Data Directory

This directory contains input videos and validation datasets used by the pipeline.

## Contents

| File / Folder | Description |
|:---|:---|
| `sample_traffic.mp4` | Intel IoT sample traffic video (768x432, 12 FPS, 647 frames) |
| `coco8/` | COCO8 mini validation dataset (auto-downloaded by Ultralytics) |

## Obtaining the Sample Video

The sample video is automatically downloaded by running:

```powershell
.venv\Scripts\python src\download_sample_data.py
```

Or it will be fetched when `src/pipeline.py` is run if the video does not exist.

## Using Your Own Video

To use a custom video, update the `VIDEO_IN_PATH` variable in `src/pipeline.py`:

```python
VIDEO_IN_PATH = "data/your_video.mp4"
```

The pipeline supports any video format readable by OpenCV (MP4, AVI, MOV, etc.).
