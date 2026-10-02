"""
YOLOv8 + Depth Anything V2 — Visualization and Analytics Generator
===================================================================
Generates publication-quality figures and sample frames for Day 3:
  1. High-resolution side-by-side comparison frames:
     [Original Video Frame | Depth Map (JET) | YOLOv8 + Depth Annotations]
  2. Analytical visualizations from detections_with_depth.csv:
     - Figure 1: Class Distribution & Average Detection Confidence
     - Figure 2: Depth Distribution Across Detected Object Classes (Box Plot)
     - Figure 3: Temporal Depth Dynamics (Object Distance vs Frame Index)
     - Figure 4: 2D Spatial Depth Heatmap (Centroid x,y colored by depth)
     - Figure 5: YOLOv8 Benchmark Performance Metrics (mAP, Precision, Recall)
     - Figure 6: Executive Analytics Dashboard (Unified 4-Panel Overview)

All figures saved to results/metrics/ and sample frames to results/sample_frames/.
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from transformers import pipeline

# ================== CONFIG ==================
VIDEO_IN_PATH = "data/sample_traffic.mp4"
VIDEO_OUT_PATH = "results/output_with_depth.mp4"
CSV_PATH = "results/detections_with_depth.csv"
METRICS_JSON_PATH = "results/metrics/evaluation_metrics.json"

SAMPLE_FRAMES_DIR = "results/sample_frames"
METRICS_DIR = "results/metrics"

DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
DEVICE = 0  # GPU index or "cpu"

KEY_FRAMES = [50, 150, 300]  # Frame indices for qualitative side-by-side figures
# =============================================

COCO_CLASSES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
    20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
    25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
    30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite",
    34: "baseball bat", 35: "baseball glove", 36: "skateboard",
    37: "surfboard", 38: "tennis racket", 39: "bottle", 40: "wine glass",
    41: "cup", 42: "fork", 43: "knife", 44: "spoon", 45: "bowl",
    46: "banana", 47: "apple", 48: "sandwich", 49: "orange", 50: "broccoli",
    51: "carrot", 52: "hot dog", 53: "pizza", 54: "donut", 55: "cake",
    56: "chair", 57: "couch", 58: "potted plant", 59: "bed",
    60: "dining table", 61: "toilet", 62: "tv", 63: "laptop", 64: "mouse",
    65: "remote", 66: "keyboard", 67: "cell phone", 68: "microwave",
    69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator", 73: "book",
    74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear",
    78: "hair drier", 79: "toothbrush"
}


def setup_style():
    """Configure modern, publication-quality plot aesthetics."""
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["axes.edgecolor"] = "#cccccc"
    plt.rcParams["axes.linewidth"] = 0.8


def extract_key_frames_and_depth():
    """Extract key frames from input and output videos, compute raw depth map, and create side-by-side comparisons."""
    print("[INFO] Loading Depth Anything V2 for sample frame visualizations...")
    depth_pipe = pipeline(task="depth-estimation", model=DEPTH_MODEL_ID, device=DEVICE)

    cap_in = cv2.VideoCapture(VIDEO_IN_PATH)
    cap_out = cv2.VideoCapture(VIDEO_OUT_PATH)

    if not cap_in.isOpened() or not cap_out.isOpened():
        print("[WARN] Could not open video files for frame extraction.")
        return

    os.makedirs(SAMPLE_FRAMES_DIR, exist_ok=True)

    for target_idx in KEY_FRAMES:
        cap_in.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
        ret_in, frame_in = cap_in.read()

        cap_out.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
        ret_out, frame_out = cap_out.read()

        if not ret_in or not ret_out:
            print(f"[WARN] Frame {target_idx} could not be read.")
            continue

        # Save single annotated frame
        single_annotated_path = os.path.join(SAMPLE_FRAMES_DIR, f"annotated_frame_{target_idx:04d}.png")
        cv2.imwrite(single_annotated_path, frame_out)

        # Compute depth map for original frame
        rgb_in = cv2.cvtColor(frame_in, cv2.COLOR_BGR2RGB)
        pil_in = Image.fromarray(rgb_in)
        depth_out = depth_pipe(pil_in)

        if isinstance(depth_out, dict) and "predicted_depth" in depth_out:
            depth_np = depth_out["predicted_depth"].squeeze().cpu().numpy()
        elif isinstance(depth_out, dict) and "depth" in depth_out:
            depth_np = np.array(depth_out["depth"], dtype=np.float32)
        else:
            depth_np = np.array(depth_out, dtype=np.float32)

        # Normalize depth map to 0-255 uint8 and apply JET colormap
        d_min, d_max = depth_np.min(), depth_np.max()
        if d_max > d_min:
            d_norm = ((depth_np - d_min) / (d_max - d_min) * 255.0).astype(np.uint8)
        else:
            d_norm = np.zeros_like(depth_np, dtype=np.uint8)

        depth_jet = cv2.applyColorMap(d_norm, cv2.COLORMAP_JET)
        depth_jet_resized = cv2.resize(depth_jet, (frame_in.shape[1], frame_in.shape[0]))

        # Save individual depth map
        depth_map_path = os.path.join(SAMPLE_FRAMES_DIR, f"depth_map_frame_{target_idx:04d}.png")
        cv2.imwrite(depth_map_path, depth_jet_resized)

        # Build Side-by-Side Comparison: [Original | Depth Map | Annotated Output]
        # Add title headers to each panel
        h, w, _ = frame_in.shape
        header_h = 40
        canvas = np.zeros((h + header_h, w * 3, 3), dtype=np.uint8)
        canvas.fill(245)

        # Place images
        canvas[header_h:, 0:w] = frame_in
        canvas[header_h:, w:2*w] = depth_jet_resized
        canvas[header_h:, 2*w:3*w] = frame_out

        # Text labels
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(canvas, "1. Original Video Frame", (20, 28), font, 0.75, (20, 20, 20), 2)
        cv2.putText(canvas, "2. Dense Depth Map (Depth Anything V2)", (w + 20, 28), font, 0.75, (20, 20, 20), 2)
        cv2.putText(canvas, "3. YOLOv8 Detections + Median Depth", (2*w + 20, 28), font, 0.75, (20, 20, 20), 2)

        sbs_path = os.path.join(SAMPLE_FRAMES_DIR, f"side_by_side_frame_{target_idx:04d}.png")
        cv2.imwrite(sbs_path, canvas)
        print(f"[OK] Generated side-by-side comparison for frame {target_idx}: {sbs_path}")

    cap_in.release()
    cap_out.release()


def plot_class_distribution(df):
    """Figure 1: Detection counts and average confidence per class."""
    plt.figure(figsize=(10, 5))
    class_counts = df["class_name"].value_counts().reset_index()
    class_counts.columns = ["class_name", "count"]

    # Merge mean confidence
    mean_conf = df.groupby("class_name")["conf"].mean().reset_index()
    merged = pd.merge(class_counts, mean_conf, on="class_name")

    ax1 = plt.gca()
    colors = sns.color_palette("Blues_r", len(merged))
    bars = ax1.bar(merged["class_name"], merged["count"], color=colors, edgecolor="black", linewidth=0.8, alpha=0.85)

    ax1.set_ylabel("Detection Count", color="#1f4e79", fontweight="bold")
    ax1.set_xlabel("Object Class", fontweight="bold")
    ax1.tick_params(axis="x", rotation=30)

    # Add count labels on bars
    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., h + 3, f"{int(h)}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Secondary axis for confidence
    ax2 = ax1.twinx()
    ax2.plot(merged["class_name"], merged["conf"], color="#e74c3c", marker="o", linewidth=2.5, markersize=8, label="Mean Confidence")
    ax2.set_ylabel("Mean Confidence Score", color="#c0392b", fontweight="bold")
    ax2.set_ylim(0.0, 1.05)
    ax2.grid(False)

    plt.title("Object Detection Frequencies & Confidence by Class", fontsize=14, fontweight="bold", pad=15)
    out_path = os.path.join(METRICS_DIR, "class_distribution.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved Figure 1: {out_path}")


def plot_depth_distribution(df):
    """Figure 2: Depth distribution box plot across detected object classes."""
    plt.figure(figsize=(11, 6))

    # Filter classes with at least 5 detections for meaningful distribution
    common_classes = df["class_name"].value_counts()[lambda x: x >= 5].index.tolist()
    sub_df = df[df["class_name"].isin(common_classes)]

    palette = sns.color_palette("Set2", len(common_classes))
    ax = sns.boxplot(
        data=sub_df,
        x="class_name",
        y="median_depth",
        palette=palette,
        fliersize=3,
        linewidth=1.2,
        boxprops=dict(alpha=0.8)
    )
    sns.stripplot(
        data=sub_df,
        x="class_name",
        y="median_depth",
        color="black",
        alpha=0.35,
        size=4,
        jitter=0.2
    )

    plt.title("Object Depth Distribution by Class (Depth Anything V2)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Detected Class", fontweight="bold")
    plt.ylabel("Relative Depth (Metric Units)", fontweight="bold")
    plt.xticks(rotation=25)

    out_path = os.path.join(METRICS_DIR, "depth_distribution_by_class.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved Figure 2: {out_path}")


def plot_temporal_depth(df):
    """Figure 3: Temporal depth dynamics showing object distances across video frames."""
    plt.figure(figsize=(12, 6))

    top_classes = df["class_name"].value_counts().head(5).index.tolist()
    filtered = df[df["class_name"].isin(top_classes)]

    palette = sns.color_palette("tab10", len(top_classes))
    sns.scatterplot(
        data=filtered,
        x="frame_idx",
        y="median_depth",
        hue="class_name",
        palette=palette,
        s=45,
        alpha=0.75,
        edgecolor="w",
        linewidth=0.5
    )

    # Add rolling median trend line for overall scene depth
    frame_medians = df.groupby("frame_idx")["median_depth"].median().rolling(window=15, min_periods=1).mean()
    plt.plot(frame_medians.index, frame_medians.values, color="black", linestyle="--", linewidth=2.0, label="Scene Depth Trend (Rolling Avg)")

    plt.title("Temporal Depth Trajectory of Detections Over Video Frames", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Video Frame Index", fontweight="bold")
    plt.ylabel("Estimated Median Depth", fontweight="bold")
    plt.legend(title="Class", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)

    out_path = os.path.join(METRICS_DIR, "depth_vs_time.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved Figure 3: {out_path}")


def plot_spatial_heatmap(df):
    """Figure 4: 2D Spatial Depth Heatmap of object bounding box centers."""
    plt.figure(figsize=(10, 6))

    cx = (df["x1"] + df["x2"]) / 2.0
    cy = (df["y1"] + df["y2"]) / 2.0

    scatter = plt.scatter(
        cx, cy,
        c=df["median_depth"],
        cmap="turbo",
        s=50,
        alpha=0.8,
        edgecolors="black",
        linewidth=0.5
    )

    cbar = plt.colorbar(scatter)
    cbar.set_label("Relative Median Depth (Lower = Closer)", fontweight="bold")

    plt.gca().invert_yaxis()  # Image coordinates: y=0 at top
    plt.title("2D Spatial Projection of Detections Colored by Depth", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Image Coordinate X (pixels)", fontweight="bold")
    plt.ylabel("Image Coordinate Y (pixels)", fontweight="bold")

    out_path = os.path.join(METRICS_DIR, "spatial_depth_heatmap.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved Figure 4: {out_path}")


def plot_yolo_benchmark():
    """Figure 5: YOLOv8 Benchmark Metrics Bar Chart."""
    if not os.path.exists(METRICS_JSON_PATH):
        print(f"[WARN] Metrics JSON not found: {METRICS_JSON_PATH}")
        return

    import json
    with open(METRICS_JSON_PATH, "r") as f:
        data = json.load(f)

    yolo = data.get("yolo_benchmark", {})
    if not yolo:
        return

    metrics_names = ["Precision", "Recall", "mAP@0.5", "mAP@0.5:0.95", "mAP@0.75"]
    metrics_vals = [
        yolo.get("precision", 0),
        yolo.get("recall", 0),
        yolo.get("mAP50", 0),
        yolo.get("mAP50_95", 0),
        yolo.get("mAP75", 0),
    ]

    plt.figure(figsize=(9, 5))
    colors = ["#2ecc71", "#3498db", "#9b59b6", "#e67e22", "#e74c3c"]
    bars = plt.bar(metrics_names, metrics_vals, color=colors, edgecolor="black", linewidth=0.8, alpha=0.85)

    plt.ylim(0.0, 1.1)
    plt.ylabel("Score [0.0 - 1.0]", fontweight="bold")
    plt.title("YOLOv8n Validation Benchmark Performance (COCO8)", fontsize=14, fontweight="bold", pad=15)

    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., h + 0.02, f"{h:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    out_path = os.path.join(METRICS_DIR, "yolo_val_performance.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved Figure 5: {out_path}")


def plot_executive_dashboard(df):
    """Figure 6: Comprehensive 4-panel Executive Dashboard."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle("YOLOv8 + Depth Anything V2: Executive Performance Dashboard", fontsize=18, fontweight="bold", y=0.99)

    # Panel 1: Top 6 classes count & depth
    ax1 = axes[0, 0]
    top_classes = df["class_name"].value_counts().head(6).index.tolist()
    sub_df = df[df["class_name"].isin(top_classes)]
    class_stats = sub_df.groupby("class_name").agg(
        count=("conf", "count"),
        mean_depth=("median_depth", "mean")
    ).loc[top_classes]

    bars = ax1.bar(class_stats.index, class_stats["count"], color="#3498db", edgecolor="black", alpha=0.8)
    ax1.set_title("A. Top Detected Classes & Counts", fontweight="bold", fontsize=13)
    ax1.set_ylabel("Detection Count", color="#2980b9", fontweight="bold")
    ax1.tick_params(axis="x", rotation=25)
    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., h + 2, f"{int(h)}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Panel 2: Depth vs Confidence Scatter
    ax2 = axes[0, 1]
    scatter = ax2.scatter(df["median_depth"], df["conf"], c=df["conf"], cmap="viridis", alpha=0.6, edgecolors="none")
    ax2.set_title("B. Detection Confidence vs Estimated Depth", fontweight="bold", fontsize=13)
    ax2.set_xlabel("Median Depth", fontweight="bold")
    ax2.set_ylabel("YOLO Confidence Score", fontweight="bold")
    fig.colorbar(scatter, ax=ax2, label="Confidence")

    # Panel 3: Depth Distribution Histogram with KDE
    ax3 = axes[1, 0]
    sns.histplot(df["median_depth"], kde=True, ax=ax3, color="#e67e22", bins=25, alpha=0.6)
    ax3.axvline(df["median_depth"].mean(), color="red", linestyle="--", linewidth=2, label=f"Mean: {df['median_depth'].mean():.2f}")
    ax3.axvline(df["median_depth"].median(), color="blue", linestyle=":", linewidth=2, label=f"Median: {df['median_depth'].median():.2f}")
    ax3.set_title("C. Scene Depth Distribution (All Objects)", fontweight="bold", fontsize=13)
    ax3.set_xlabel("Relative Depth Units", fontweight="bold")
    ax3.set_ylabel("Frequency", fontweight="bold")
    ax3.legend(frameon=True)

    # Panel 4: Spatial Centroid Map
    ax4 = axes[1, 1]
    cx = (df["x1"] + df["x2"]) / 2.0
    cy = (df["y1"] + df["y2"]) / 2.0
    sc = ax4.scatter(cx, cy, c=df["median_depth"], cmap="coolwarm", s=35, alpha=0.75, edgecolors="black", linewidth=0.3)
    ax4.invert_yaxis()
    ax4.set_title("D. 2D Spatial Centroid Heatmap", fontweight="bold", fontsize=13)
    ax4.set_xlabel("Image X (px)", fontweight="bold")
    ax4.set_ylabel("Image Y (px)", fontweight="bold")
    fig.colorbar(sc, ax=ax4, label="Depth (Closer = Blue, Farther = Red)")

    plt.tight_layout()
    out_path = os.path.join(METRICS_DIR, "executive_dashboard.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved Figure 6: {out_path}")


def main():
    print("=" * 60)
    print("DAY 3: VISUALIZATION & ANALYTICS GENERATION")
    print("=" * 60)

    setup_style()

    # 1. Load CSV data
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Detections CSV not found: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    df["class_name"] = df["class_id"].map(lambda cid: COCO_CLASSES.get(int(cid), f"class_{cid}"))
    print(f"[INFO] Loaded {len(df)} detections from {CSV_PATH}")

    # 2. Generate side-by-side key frames
    extract_key_frames_and_depth()

    # 3. Generate analytical plots
    plot_class_distribution(df)
    plot_depth_distribution(df)
    plot_temporal_depth(df)
    plot_spatial_heatmap(df)
    plot_yolo_benchmark()
    plot_executive_dashboard(df)

    print("=" * 60)
    print("DAY 3 VISUALIZATIONS GENERATED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()
