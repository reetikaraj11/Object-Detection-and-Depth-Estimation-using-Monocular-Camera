# IUI Midterm Report — Object Detection & Depth Estimation using Monocular Camera

## YOLOv8 + Depth Anything V2: A Hybrid Computer Vision Pipeline

**Course:** AI for Design & Manufacturing (AI4DM)  
**Institute:** Indian Institute of Science, Bangalore — I3D Lab  
**Semester:** Sem-3, 2026  

---

## 1. Introduction

### 1.1 Problem Statement

Human visual perception excels at straight-forward distance estimation. However, when operating through a monocular camera — as is common in drone, robotic, and surveillance applications — depth perception is fundamentally lost. Standard 2D object detection models like YOLO can identify *what* objects are present and *where* they are in the image plane, but cannot determine *how far* they are from the camera.

This creates an unacceptable limitation for remote drone operations, autonomous navigation, and robotic manipulation tasks where metric distance measurement is essential for reliable operation.

### 1.2 Objective

We implement a **hybrid YOLOv8 + Depth Anything V2 (DA-V2) architecture** that combines:
1. **YOLOv8** — Fast, accurate real-time object detection (bounding boxes, class labels, confidence scores)
2. **Depth Anything V2** — State-of-the-art monocular depth estimation (dense per-pixel relative depth maps)

Together, these models provide both object identification and spatial distance awareness from a single monocular camera input — enabling applications such as drone navigation, obstacle avoidance, and robotic pick-and-place without requiring specialized depth sensors (LiDAR, stereo cameras).

### 1.3 Why This Matters

| Capability | YOLO Alone | YOLO + DA-V2 |
|:---|:---:|:---:|
| Object Detection (class, bbox) | Yes | Yes |
| Confidence Score | Yes | Yes |
| **Depth / Distance Estimation** | **No** | **Yes** |
| **3D Spatial Awareness** | **No** | **Yes** |
| **Inter-Object Distance** | **No** | **Yes** |

---

## 2. Background

### 2.1 YOLOv8 Architecture

YOLOv8 (Ultralytics, 2023) is a state-of-the-art single-stage object detection model built on the CSP-Darknet backbone architecture. Key features:
- **Single-pass detection**: Predicts bounding boxes and class probabilities in one forward pass
- **Anchor-free design**: Eliminates anchor box hyperparameter tuning
- **Multi-scale feature fusion**: FPN + PAN neck for detecting objects at multiple scales
- **Pre-trained on COCO**: 80-class detection with strong generalization

We use **YOLOv8n** (nano variant) — the smallest and fastest variant with 3.15M parameters, suitable for our GTX 1650 GPU (4 GB VRAM).

### 2.2 Depth Anything V2

Depth Anything V2 (Yang et al., 2024) is a foundation model for monocular depth estimation that stands out for its:
- **Zero-shot capability**: Trained on 150M+ unlabeled images via self-supervised learning; generalizes to any scene without fine-tuning
- **Overcoming MiDaS limitations**: Addresses the data coverage problem of earlier foundation MDE models by leveraging massive unlabeled datasets
- **Dense depth maps**: Produces per-pixel relative depth for the entire frame
- **Multiple model scales**: Available in Small (24.8M), Base (97.5M), and Large (335.3M) variants

We use the **Small** variant (`depth-anything/Depth-Anything-V2-Small-hf`) via HuggingFace Transformers, compatible with our 4 GB VRAM constraint.

### 2.3 Why a Hybrid Pipeline Approach

Rather than modifying model internals (as required by the DINOv2 fusion approach), our pipeline architecture uses both models **as-is** in sequence:

```
Video Frame --> YOLOv8 (detection) --> Bounding Boxes + Class + Confidence
     |
     +-------> Depth Anything V2 ----> Dense Depth Map (HxW float32)
                                            |
                    For each bbox: median depth from central window
                                            |
                    Annotated frame (bboxes + depth labels + depth thumbnail)
                                            |
                    Output: annotated .mp4 + detections .csv
```

**Advantages:**
- No source code modification required
- Both models work independently — lower debugging risk
- Easy to swap model variants or upgrade individually
- More visually impressive outputs (depth maps + annotated detections)

---

## 3. Dataset Description

### 3.1 COCO8 Validation Dataset
- **Purpose**: Standard benchmark for evaluating YOLOv8 detection accuracy
- **Source**: Built into Ultralytics (`coco8.yaml`)
- **Size**: 4 validation images, 17 object instances
- **Classes**: Standard COCO 80-class taxonomy

### 3.2 Video Dataset for Pipeline Demo
- **Source**: Intel IoT DevKit sample traffic video
- **Format**: MP4, H.264 encoded
- **Resolution**: 768 x 432 pixels
- **Frame Rate**: 12 FPS
- **Total Frames**: 647
- **Duration**: ~54 seconds
- **Content**: Urban traffic scene with pedestrians, cyclists, vehicles at varying distances

---

## 4. Methodology

### 4.1 System Architecture (Algorithm 1)

The pipeline implements the complete CV framework described in Algorithm 1:

**Step 1 — Object Detection (YOLOv8):**
For each video frame $I_t$ at timestep $t$:
- Run YOLOv8 inference to obtain bounding boxes: $[x_1, y_1, x_2, y_2]$ with class ID and confidence score
- Apply confidence threshold (0.25) to filter low-quality detections

**Step 2 — Depth Map Generation (Depth Anything V2):**
- Convert the same frame to RGB PIL Image
- Run DA-V2 inference to obtain dense depth map $D$ of dimension $H \times W$
- Depth values represent relative metric depth (lower = closer, higher = farther)

**Step 3 — Per-Object Distance Estimation:**
- For each detected bounding box, compute the **midpoint anchor point**:

$$cx = \frac{x_1 + x_2}{2}, \quad cy = \frac{y_1 + y_2}{2}$$

- Extract **median depth** from a central $5 \times 5$ pixel window within the bounding box for robustness against boundary contamination:

$$d_{obj} = \text{median}(D[cy-2:cy+3, cx-2:cx+3])$$

**Step 4 — 3D Camera Coordinate Projection:**
Using intrinsic camera parameters $K = [f_x, f_y, c_{x0}, c_{y0}]$:

$$X = \frac{(cx - c_{x0}) \cdot d}{f_x}, \quad Y = \frac{(cy - c_{y0}) \cdot d}{f_y}, \quad Z = d$$

**Step 5 — Inter-Object Distance (Law of Cosines):**
For any two detected objects with depths $d_1$ and $d_2$:

$$\theta_i = \arctan\left(\frac{cx_i - c_{x0}}{f_x}\right), \quad \Delta\theta = |\theta_1 - \theta_2|$$

$$D_{\text{cosines}} = \sqrt{d_1^2 + d_2^2 - 2 d_1 d_2 \cos(\Delta\theta)}$$

$$D_{3D} = \sqrt{(X_1 - X_2)^2 + (Y_1 - Y_2)^2 + (Z_1 - Z_2)^2}$$

### 4.2 Key Design Decisions

| Decision | Rationale |
|:---|:---|
| **Central 5x5 window** for depth sampling | Suppresses background boundary bleed from bounding box edges |
| **Median** aggregation (not mean) | Robust to depth outliers caused by occlusion or edge artifacts |
| **YOLOv8 nano** variant | Optimized for 4 GB VRAM constraint while maintaining strong detection |
| **DA-V2 Small** variant | Balances depth quality and inference speed for real-time feasibility |

---

## 5. Utility of the Pre-Trained Model (Depth Anything V2)

> **This section addresses Mail-2 requirement: "Explain the utility of the pre-trained model."**

### 5.1 Zero-Shot Transfer Learning

Depth Anything V2 was trained on **150M+ unlabeled images** via self-supervised learning. Unlike supervised depth models that require expensive ground-truth depth annotations (from LiDAR or structured light), DA-V2 learns depth representations from the natural structure of images. This means:

- **No fine-tuning needed** for our traffic video — the model generalizes out-of-the-box
- **No depth-labeled data required** for any specific deployment scenario
- **Works on arbitrary scenes** — indoor, outdoor, aerial, underwater

### 5.2 Solving the Fundamental Gap

YOLO tells us **what** and **where** (in 2D) an object is, but not **how far** it is. This is insufficient for:
- **Drone navigation**: A drone must know whether an obstacle is 2m or 20m away
- **Robotic manipulation**: A pick-and-place robot needs metric depth to plan arm trajectories
- **ADAS (Advanced Driver Assistance)**: Collision warning systems require distance estimation

DA-V2 provides the missing **third dimension** — converting 2D detections into spatially-aware 3D scene understanding.

### 5.3 Overcoming MiDaS Limitations

Earlier foundation depth models like MiDaS suffered from **data coverage problems** — they were trained on limited labeled datasets, leading to poor generalization in unseen environments. DA-V2 specifically addresses this by:
1. Leveraging **unlabeled** image datasets at massive scale
2. Using a **teacher-student** self-supervised training paradigm
3. Achieving **consistent relative depth ordering** even in complex, cluttered scenes

### 5.4 Practical Applications

| Application | How DA-V2 Helps |
|:---|:---|
| **Drone Navigation** | Estimates obstacle distance from a single camera — no LiDAR needed |
| **Robotic Pick-and-Place** | Provides depth for trajectory planning using only a monocular camera |
| **Autonomous Vehicles** | Supplements LiDAR/radar with dense visual depth in blind spots |
| **Augmented Reality** | Enables realistic 3D object placement in monocular video |
| **Surveillance** | Estimates proximity of detected persons/vehicles for spatial analysis |

---

## 6. Results

### 6.1 YOLOv8n Detection Benchmark (COCO8)

| Metric | Value | Description |
|:---|:---|:---|
| **mAP@0.5** | 0.8875 (88.75%) | Mean Average Precision at IoU=0.50 |
| **mAP@0.5:0.95** | 0.6291 (62.91%) | Primary COCO challenge metric |
| **mAP@0.75** | 0.6347 (63.47%) | High-precision localization |
| **Precision** | 0.6210 (62.10%) | True positive ratio |
| **Recall** | 0.8333 (83.33%) | Detection coverage |
| **Inference Latency** | 11.70 ms | GPU inference time per image |
| **Total Latency** | 15.95 ms | End-to-end (preprocess + inference + postprocess) |

### 6.2 Pipeline Execution Statistics

| Metric | Value |
|:---|:---|
| **Input Video** | `sample_traffic.mp4` (768x432, 12 FPS, 647 frames) |
| **Processing Speed** | ~4.2 FPS on NVIDIA GTX 1650 |
| **Total Detections** | 421 |
| **Frames with Detections** | 265 / 647 (40.9%) |
| **Unique Classes Detected** | 11 |
| **Confidence Range** | [0.253, 0.950], Mean = 0.651, Median = 0.705 |
| **Depth Range** | [0.97, 9.84] relative units |
| **Mean Depth** | 4.14 |
| **Median Depth** | 3.09 |

### 6.3 Per-Class Detection & Depth Analysis

| Class | Count | Share (%) | Mean Confidence | Mean Depth | Depth Range |
|:---|---:|---:|---:|---:|:---|
| person | 226 | 53.68% | 0.672 | 3.76 | [0.97, 9.84] |
| car | 84 | 19.95% | 0.690 | 5.26 | [3.44, 7.87] |
| bicycle | 62 | 14.73% | 0.682 | 2.38 | [1.76, 2.68] |
| cell phone | 16 | 3.80% | 0.587 | 6.17 | [4.88, 7.16] |
| kite | 13 | 3.09% | 0.356 | 7.47 | [7.41, 7.51] |
| boat | 7 | 1.66% | 0.490 | 7.49 | [7.37, 7.57] |
| motorcycle | 3 | 0.71% | 0.304 | 2.26 | [2.19, 2.35] |
| bus | 3 | 0.71% | 0.361 | 6.75 | [6.04, 7.41] |
| tennis racket | 3 | 0.71% | 0.388 | 1.82 | [1.34, 2.15] |
| sports ball | 2 | 0.48% | 0.395 | 8.10 | [7.88, 8.32] |
| skateboard | 2 | 0.48% | 0.272 | 2.44 | [2.35, 2.53] |

### 6.4 Inter-Object Spatial Distance Analysis

Using Algorithm 1's Law of Cosines and 3D Euclidean distance formulas:
- **Total pairwise measurements**: 208 object pairs across 110 multi-object frames
- **Distance methods validated**: Law of Cosines and full 3D Cartesian distances show strong agreement

### 6.5 Visualization Summary

All visualizations generated at 300 DPI resolution:

1. **Side-by-Side Comparisons** (3 key frames): Original Video | Dense Depth Map (JET) | YOLOv8 Annotations + Depth
2. **Class Distribution Chart**: Detection counts and mean confidence per class
3. **Depth Distribution Box Plot**: Depth spread across object classes
4. **Temporal Depth Trajectory**: Object depth evolution over video frames
5. **2D Spatial Depth Heatmap**: Centroid positions colored by estimated depth
6. **YOLOv8 Benchmark Bar Chart**: mAP, Precision, Recall comparison
7. **Executive Dashboard**: Unified 4-panel analytical overview
8. **Distance-Annotated Frame**: Algorithm 1 in action with inter-object distance vectors
9. **Distance Correlation Analysis**: Law of Cosines vs 3D Euclidean scatter + pair-type box plots

---

## 7. Analysis & Discussion

### 7.1 Spatial Depth Stratification

The pipeline demonstrates strong spatial consistency:
- **Foreground objects** (bicycles at mean depth 2.38, nearby pedestrians at median depth 2.81) correctly register lower relative depth values
- **Mid-range objects** (cars at mean depth 5.26) occupy intermediate depth bands
- **Background objects** (buses at mean depth 6.75, boats at 7.49) correctly register higher depths

This validates DA-V2's ability to produce semantically and geometrically consistent depth orderings.

### 7.2 Detection Confidence vs. Depth

Analysis shows that detection confidence does not strongly correlate with object depth — YOLO maintains reliable detection quality across the depth range. This is important because it means the pipeline provides consistent spatial awareness regardless of distance.

### 7.3 Central Window Robustness

Using a central $5 \times 5$ pixel window for depth sampling (rather than the full bounding box) effectively eliminates background boundary contamination. This technique is particularly important for small objects where the bounding box may include significant background pixels.

### 7.4 Limitations

1. **Relative vs. Absolute Depth**: DA-V2 produces relative depth values, not absolute metric distances (meters). Camera calibration or scale recovery is needed for true metric depth.
2. **Inference Speed**: At ~4.2 FPS, the pipeline is not yet real-time. DA-V2 is the primary bottleneck (~100-200ms per frame). Frame-skipping or model distillation could improve throughput.
3. **Occlusion**: Partially occluded objects may have unreliable depth estimates due to depth map blending at occlusion boundaries.
4. **Small Objects**: Very small distant objects produce tiny bounding boxes where the $5 \times 5$ central window may not have enough valid depth pixels.

### 7.5 Performance Feasibility

| Component | Latency | Throughput |
|:---|:---|:---|
| YOLOv8n (detection) | 11.70 ms | ~85 FPS |
| DA-V2 Small (depth) | ~200 ms | ~5 FPS |
| Full pipeline | ~238 ms | ~4.2 FPS |

YOLOv8 inference is extremely fast. The depth estimation model is the primary bottleneck. For real-time applications, strategies include:
- Processing depth every $N$th frame and interpolating
- Using the DA-V2 Small model (already done)
- ONNX/TensorRT optimization
- Asynchronous depth processing on a separate GPU stream

---

## 8. Conclusion & Future Work

### 8.1 Summary

We successfully implemented a **hybrid YOLOv8 + Depth Anything V2 pipeline** that:
1. Detects objects with **88.75% mAP@0.5** accuracy
2. Estimates relative depth for every detected object
3. Computes inter-object 3D spatial distances using Algorithm 1
4. Produces publication-quality annotated video, CSV logs, and analytical visualizations
5. Runs entirely on a consumer GPU (NVIDIA GTX 1650, 4 GB VRAM)

The pre-trained Depth Anything V2 model adds critical spatial awareness that YOLOv8 alone cannot provide — enabling applications in drone navigation, robotics, and autonomous systems.

### 8.2 Future Work

1. **Metric Depth Calibration**: Integrate camera intrinsics from calibration to convert relative depth to absolute metric distances
2. **Real-Time Optimization**: Deploy DA-V2 via ONNX Runtime or TensorRT for 10x+ speedup
3. **Multi-Object Tracking**: Add object tracking (ByteTrack / BoT-SORT) to maintain consistent depth trajectories per object over time
4. **Custom Dataset Fine-Tuning**: Fine-tune YOLOv8 on domain-specific objects (e.g., industrial equipment, drone targets)
5. **Stereo Validation**: Compare monocular depth estimates against stereo camera ground truth to quantify accuracy

---

## 9. References

1. Ultralytics, "YOLOv8," 2023. [Online]. Available: https://github.com/ultralytics/ultralytics
2. L. Yang et al., "Depth Anything V2," arXiv:2406.09414, 2024.
3. R. Ranftl et al., "Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-Shot Cross-Dataset Transfer," IEEE TPAMI, 2022.
4. M. Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision," arXiv:2304.07193, 2023.
5. COCO Dataset. [Online]. Available: https://cocodataset.org
6. Professor Pradipta Biswas, "AI Resource for Design & Manufacturing," IISc I3D Lab. [Online]. Available: https://cambum.net/I3DLab/AI4DM.htm

---

## 10. Appendix

### A. Project Structure

```
IUI/
├── .venv/                          # Python virtual environment (uv)
├── data/
│   ├── sample_traffic.mp4          # Input video (768x432, 647 frames)
│   └── coco8/                      # COCO8 validation dataset
├── src/
│   ├── pipeline.py                 # Main YOLOv8 + DA-V2 pipeline
│   ├── evaluate.py                 # YOLO validation + CSV analysis
│   ├── visualize.py                # Publication-quality figure generation
│   ├── spatial_distance.py         # Algorithm 1: 3D distance estimation
│   ├── verify_all_days.py          # End-to-end audit suite
│   ├── check_env.py                # Environment verification
│   └── download_sample_data.py     # Sample data downloader
├── notebooks/
│   └── analysis.ipynb              # Interactive analysis notebook
├── results/
│   ├── output_with_depth.mp4       # Annotated output video
│   ├── detections_with_depth.csv   # Detection + depth CSV log
│   ├── inter_object_distances.csv  # 3D inter-object distances
│   ├── metrics/                    # Evaluation figures + JSON
│   └── sample_frames/             # Side-by-side + annotated frames
├── report/
│   └── IUI_Midterm_Report.md       # This report
├── weights/
│   └── yolov8n.pt                  # YOLOv8 nano weights
├── requirements.txt
├── README.md
└── .gitignore
```

### B. Hardware & Software Configuration

| Component | Specification |
|:---|:---|
| **GPU** | NVIDIA GeForce GTX 1650 (4 GB VRAM) |
| **Python** | 3.12.12 |
| **PyTorch** | 2.6.0+cu124 |
| **Ultralytics** | 8.4.157 |
| **Transformers** | 5.17.0 |
| **OpenCV** | 4.11.0 |
| **OS** | Windows 10/11 |

### C. How to Reproduce

```powershell
# 1. Create and activate virtual environment
uv venv .venv
.venv\Scripts\Activate.ps1

# 2. Install dependencies
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
uv pip install -r requirements.txt

# 3. Run pipeline
.venv\Scripts\python src\pipeline.py

# 4. Run evaluation
.venv\Scripts\python src\evaluate.py

# 5. Generate visualizations
.venv\Scripts\python src\visualize.py

# 6. Run 3D distance estimation
.venv\Scripts\python src\spatial_distance.py

# 7. Verify all deliverables
.venv\Scripts\python src\verify_all_days.py
```
