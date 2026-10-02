# Object Detection & Depth Estimation using Monocular Camera

## YOLOv8 + Depth Anything V2: A Hybrid Computer Vision Pipeline

**Course:** AI for Design & Manufacturing (AI4DM)  
**Institute:** Indian Institute of Science, Bangalore — I3D Lab  
**Semester:** Sem-3, 2026  

---

## Project Overview

This project implements a **hybrid YOLOv8 + Depth Anything V2 (DA-V2) pipeline** for simultaneous object detection and monocular depth estimation. The system:

1. **Detects objects** (vehicles, pedestrians, cyclists, etc.) using YOLOv8n (pre-trained on COCO)
2. **Estimates dense depth maps** using Depth Anything V2 (self-supervised, zero-shot)
3. **Computes per-object median depth** by projecting bounding boxes onto the depth map
4. **Calculates 3D inter-object distances** using Algorithm 1 (camera intrinsics + Law of Cosines)
5. **Generates publication-quality visualizations** and analytical plots

### Why Depth Anything V2?

| Capability | YOLO Alone | YOLO + DA-V2 |
|:---|:---:|:---:|
| Object Detection (class, bbox) | Yes | Yes |
| Depth / Distance Estimation | **No** | **Yes** |
| 3D Spatial Awareness | **No** | **Yes** |
| Inter-Object Distance | **No** | **Yes** |

---

## Project Structure

```
IUI/
├── data/
│   ├── sample_traffic.mp4          # Input video (768x432, 647 frames)
│   └── coco8/                      # COCO8 validation dataset
├── src/
│   ├── pipeline.py                 # Main YOLOv8 + DA-V2 pipeline
│   ├── evaluate.py                 # YOLO validation + CSV analysis
│   ├── visualize.py                # Publication-quality figure generation
│   ├── spatial_distance.py         # Algorithm 1: 3D distance estimation
│   ├── verify_all_days.py          # End-to-end audit suite
│   ├── generate_notebook.py        # Jupyter notebook generator
│   ├── check_env.py                # Environment verification
│   └── download_sample_data.py     # Sample data downloader
├── notebooks/
│   └── analysis.ipynb              # Interactive analysis notebook (32 cells)
├── results/
│   ├── output_with_depth.mp4       # Annotated output video (3.2 MB)
│   ├── detections_with_depth.csv   # Detection + depth CSV (421 detections)
│   ├── inter_object_distances.csv  # 3D inter-object distances (208 pairs)
│   ├── metrics/                    # Evaluation figures + JSON + markdown
│   │   ├── evaluation_metrics.json
│   │   ├── evaluation_summary.md
│   │   ├── class_distribution.png
│   │   ├── depth_distribution_by_class.png
│   │   ├── depth_vs_time.png
│   │   ├── spatial_depth_heatmap.png
│   │   ├── yolo_val_performance.png
│   │   ├── executive_dashboard.png
│   │   └── distance_estimation_analysis.png
│   └── sample_frames/             # Side-by-side + annotated + depth map frames
├── report/
│   └── IUI_Midterm_Report.md       # Comprehensive report
├── weights/
│   └── yolov8n.pt                  # YOLOv8 nano weights (6.2 MB)
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Quick Start

### 1. Environment Setup (Windows / PowerShell)

```powershell
# Create and activate virtual environment
uv venv .venv
.venv\Scripts\Activate.ps1

# Install PyTorch with CUDA 12.4
uv pip install --link-mode=copy torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install all dependencies
uv pip install --link-mode=copy -r requirements.txt
```

### 2. Verify Environment

```powershell
.venv\Scripts\python src\check_env.py
```

### 3. Run Pipeline

Processes the input video through YOLOv8 + Depth Anything V2:

```powershell
.venv\Scripts\python src\pipeline.py
```

**Outputs:**
- `results/output_with_depth.mp4` — Annotated video with bboxes + depth labels + depth map overlay
- `results/detections_with_depth.csv` — Per-detection CSV log

### 4. Run Evaluation

Validates YOLOv8 on COCO8 and analyzes detection/depth statistics:

```powershell
.venv\Scripts\python src\evaluate.py
```

### 5. Generate Visualizations

Creates 6 publication-quality analytical figures + side-by-side comparison frames:

```powershell
.venv\Scripts\python src\visualize.py
```

### 6. Run 3D Distance Estimation

Implements Algorithm 1 — inter-object spatial distance using Law of Cosines:

```powershell
.venv\Scripts\python src\spatial_distance.py
```

### 7. Verify All Deliverables

Automated audit suite checking every deliverable from Days 1–4:

```powershell
.venv\Scripts\python src\verify_all_days.py
```

### 8. Open Jupyter Notebook

```powershell
.venv\Scripts\jupyter notebook notebooks\analysis.ipynb
```

---

## Key Results

| Metric | Value |
|:---|:---|
| **mAP@0.5** | 88.75% |
| **mAP@0.5:0.95** | 62.91% |
| **Precision** | 62.10% |
| **Recall** | 83.33% |
| **Total Detections** | 421 (11 COCO classes) |
| **Depth Range** | [0.97, 9.84] relative units |
| **Processing Speed** | ~4.2 FPS on GTX 1650 |
| **Inter-Object Distances** | 208 pairs across 110 frames |

---

## Hardware & Software

| Component | Specification |
|:---|:---|
| **GPU** | NVIDIA GeForce GTX 1650 (4 GB VRAM) |
| **Python** | 3.12.12 |
| **PyTorch** | 2.6.0+cu124 |
| **Ultralytics** | 8.4.157 |
| **Transformers** | 5.17.0 |
| **OpenCV** | 4.11.0 |

---

## References

1. Ultralytics YOLOv8 — https://github.com/ultralytics/ultralytics
2. Depth Anything V2 (Yang et al., 2024) — arXiv:2406.09414
3. AI4DM Course — https://cambum.net/I3DLab/AI4DM.htm
