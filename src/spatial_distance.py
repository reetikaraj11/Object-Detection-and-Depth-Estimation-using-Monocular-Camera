"""
System 2 — 3D Spatial Distance Estimation (Algorithm 1)
========================================================
Implements the exact mathematical framework and Algorithm 1 from pages 10–12 
of the course PDF ("Object Detection and Depth Estimation using Monocular Camera"):

Algorithm Steps:
  1. For each video frame:
     - Detect bounding boxes of objects [x1, y1, x2, y2] using YOLOv8
     - Compute midpoint anchor points: cx = (x1 + x2) / 2, cy = (y1 + y2) / 2
     - Project anchor points onto the Depth Anything V2 depth map to get d1, d2
  2. Camera Coordinate Conversion:
     - Using camera intrinsic matrix K:
       X = (cx - cx0) * d / fx
       Y = (cy - cy0) * d / fy
       Z = d
  3. Angle & Distance Calculation:
     - Compute azimuth angle: theta = arctan((cx - cx0) / fx)
     - Compute angular separation: delta_theta = |theta1 - theta2|
     - Apply Law of Cosines:
       D_cosines = sqrt(d1^2 + d2^2 - 2 * d1 * d2 * cos(delta_theta))
     - Full 3D Euclidean distance:
       D_3D = sqrt((X1 - X2)^2 + (Y1 - Y2)^2 + (Z1 - Z2)^2)
  4. Track inter-object spatial distances (e.g. person-to-bicycle, vehicle-to-vehicle, drone-to-target)
  5. Generate distance-annotated frames, CSV log, and analytical plots.

Outputs:
  - results/inter_object_distances.csv
  - results/metrics/distance_estimation_analysis.png
  - results/sample_frames/distance_annotated_frame.png
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ================== CONFIG ==================
CSV_PATH = "results/detections_with_depth.csv"
VIDEO_IN_PATH = "data/sample_traffic.mp4"
METRICS_DIR = "results/metrics"
SAMPLE_FRAMES_DIR = "results/sample_frames"
OUTPUT_CSV = "results/inter_object_distances.csv"
OUTPUT_PLOT = os.path.join(METRICS_DIR, "distance_estimation_analysis.png")
OUTPUT_FRAME = os.path.join(SAMPLE_FRAMES_DIR, "distance_annotated_frame.png")

# Camera Intrinsics K for video resolution 768x432 (standard pinhole approximation)
# fx, fy ~ 0.8 * image_width (standard field of view ~ 64 deg horizontal)
IMG_W, IMG_H = 768, 432
FX = 0.8 * IMG_W   # ~614.4 px
FY = 0.8 * IMG_W   # square pixels assumption
CX0 = IMG_W / 2.0  # 384.0 px (principal point x)
CY0 = IMG_H / 2.0  # 216.0 px (principal point y)

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


def image_to_camera_coordinates(cx, cy, depth, fx=FX, fy=FY, cx0=CX0, cy0=CY0):
    """
    Converts 2D image coordinates (cx, cy) and depth (d) to 3D camera coordinates (X, Y, Z).
    As described in Algorithm 1, Step 3.
    """
    X = (cx - cx0) * depth / fx
    Y = (cy - cy0) * depth / fy
    Z = depth
    return X, Y, Z


def compute_angle_and_distance(obj1, obj2, fx=FX, cx0=CX0):
    """
    Computes azimuth angle, Law of Cosines distance, and 3D Euclidean distance
    between two detected objects per PDF page 10-11 formulas.
    """
    # Midpoints (anchor points)
    cx1, cy1 = (obj1["x1"] + obj1["x2"]) / 2.0, (obj1["y1"] + obj1["y2"]) / 2.0
    cx2, cy2 = (obj2["x1"] + obj2["x2"]) / 2.0, (obj2["y1"] + obj2["y2"]) / 2.0

    d1, d2 = obj1["median_depth"], obj2["median_depth"]

    # Azimuth angles in camera frame: theta = arctan((cx - cx0) / fx)
    theta1 = np.arctan((cx1 - cx0) / fx)
    theta2 = np.arctan((cx2 - cx0) / fx)
    delta_theta = np.abs(theta1 - theta2)

    # Law of Cosines distance: D = sqrt(d1^2 + d2^2 - 2*d1*d2*cos(delta_theta))
    d_cosines = np.sqrt(max(0.0, d1**2 + d2**2 - 2 * d1 * d2 * np.cos(delta_theta)))

    # Full 3D Camera Coordinates
    X1, Y1, Z1 = image_to_camera_coordinates(cx1, cy1, d1)
    X2, Y2, Z2 = image_to_camera_coordinates(cx2, cy2, d2)
    d_3d = np.sqrt((X1 - X2)**2 + (Y1 - Y2)**2 + (Z1 - Z2)**2)

    return {
        "cx1": cx1, "cy1": cy1, "d1": d1, "theta1_deg": np.degrees(theta1),
        "cx2": cx2, "cy2": cy2, "d2": d2, "theta2_deg": np.degrees(theta2),
        "delta_theta_deg": np.degrees(delta_theta),
        "distance_cosines": float(d_cosines),
        "distance_3d": float(d_3d),
        "X1": X1, "Y1": Y1, "Z1": Z1,
        "X2": X2, "Y2": Y2, "Z2": Z2,
    }


def analyze_inter_object_distances():
    """Extract pair-wise distances for all frames where multiple objects are present."""
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Detections CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    df["class_name"] = df["class_id"].map(lambda cid: COCO_CLASSES.get(int(cid), f"class_{cid}"))

    # Group by frame
    records = []
    for frame_idx, group in df.groupby("frame_idx"):
        if len(group) < 2:
            continue  # Need at least 2 objects to compute distance

        objs = group.to_dict("records")
        # Pairwise combination of objects
        for i in range(len(objs)):
            for j in range(i + 1, len(objs)):
                o1 = objs[i]
                o2 = objs[j]

                dist_info = compute_angle_and_distance(o1, o2)
                records.append({
                    "frame_idx": int(frame_idx),
                    "obj1_idx": int(o1["det_idx"]),
                    "obj1_class": o1["class_name"],
                    "obj1_conf": float(o1["conf"]),
                    "obj2_idx": int(o2["det_idx"]),
                    "obj2_class": o2["class_name"],
                    "obj2_conf": float(o2["conf"]),
                    "pair_type": f"{o1['class_name']}-{o2['class_name']}",
                    "d1": dist_info["d1"],
                    "d2": dist_info["d2"],
                    "delta_theta_deg": dist_info["delta_theta_deg"],
                    "distance_cosines": dist_info["distance_cosines"],
                    "distance_3d": dist_info["distance_3d"],
                    "cx1": dist_info["cx1"], "cy1": dist_info["cy1"],
                    "cx2": dist_info["cx2"], "cy2": dist_info["cy2"],
                })

    dist_df = pd.DataFrame(records)
    dist_df.to_csv(OUTPUT_CSV, index=False)
    print(f"[OK] Computed {len(dist_df)} inter-object distance measurements across {dist_df['frame_idx'].nunique()} multi-object frames.")
    print(f"[OK] Saved inter-object distances to: {OUTPUT_CSV}")
    return dist_df


def render_distance_annotated_frame(dist_df):
    """Render a representative video frame illustrating Algorithm 1 with distance vectors."""
    if dist_df.empty:
        print("[WARN] No multi-object frames to annotate.")
        return

    # Find a frame with prominent detections and clean distance measurement (e.g. frame ~50 or with person-bicycle/car)
    candidate_frames = dist_df["frame_idx"].value_counts().index.tolist()
    chosen_frame = candidate_frames[0]
    for fid in candidate_frames:
        f_rows = dist_df[dist_df["frame_idx"] == fid]
        # Prefer person-car or person-bicycle pairs
        if any("person" in p for p in f_rows["pair_type"]):
            chosen_frame = fid
            break

    frame_rows = dist_df[dist_df["frame_idx"] == chosen_frame]
    print(f"[INFO] Annotating frame {chosen_frame} with {len(frame_rows)} spatial distance vectors...")

    cap = cv2.VideoCapture(VIDEO_IN_PATH)
    cap.set(cv2.CAP_PROP_POS_FRAMES, chosen_frame)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"[WARN] Could not read frame {chosen_frame} from {VIDEO_IN_PATH}")
        return

    annotated = frame.copy()
    overlay = annotated.copy()

    # Draw distance vectors
    for _, row in frame_rows.iterrows():
        pt1 = (int(row["cx1"]), int(row["cy1"]))
        pt2 = (int(row["cx2"]), int(row["cy2"]))

        # Draw anchor points (cyan circles)
        cv2.circle(annotated, pt1, 6, (255, 255, 0), -1)
        cv2.circle(annotated, pt1, 8, (0, 0, 0), 2)
        cv2.circle(annotated, pt2, 6, (255, 255, 0), -1)
        cv2.circle(annotated, pt2, 8, (0, 0, 0), 2)

        # Draw connecting line (dashed/yellow)
        cv2.line(overlay, pt1, pt2, (0, 215, 255), 3, cv2.LINE_AA)

        # Distance text at midpoint
        mid_x = (pt1[0] + pt2[0]) // 2
        mid_y = (pt1[1] + pt2[1]) // 2 - 10
        dist_text = f"Dist: {row['distance_3d']:.2f}m (d_theta:{row['delta_theta_deg']:.1f}deg)"
        (tw, th), _ = cv2.getTextSize(dist_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (mid_x - 4, mid_y - th - 4), (mid_x + tw + 4, mid_y + 4), (0, 0, 0), -1)
        cv2.putText(annotated, dist_text, (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    # Blend overlay
    cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)

    # Add header banner
    cv2.rectangle(annotated, (0, 0), (IMG_W, 35), (20, 20, 20), -1)
    cv2.putText(annotated, f"Algorithm 1: 3D Spatial Distance Estimation (Frame {chosen_frame})", (15, 24),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 255, 255), 1, cv2.LINE_AA)

    os.makedirs(SAMPLE_FRAMES_DIR, exist_ok=True)
    cv2.imwrite(OUTPUT_FRAME, annotated)
    print(f"[OK] Saved distance-annotated frame: {OUTPUT_FRAME}")


def plot_distance_analysis(dist_df):
    """Generate analytical figures showing inter-object distance distributions and dynamics."""
    if dist_df.empty:
        return

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Algorithm 1: 3D Inter-Object Spatial Distance Analysis", fontsize=15, fontweight="bold")

    # Panel 1: Law of Cosines vs Full 3D Euclidean Distance correlation
    ax1 = axes[0]
    ax1.scatter(dist_df["distance_cosines"], dist_df["distance_3d"], color="#2980b9", alpha=0.6, edgecolors="none")
    max_val = max(dist_df["distance_cosines"].max(), dist_df["distance_3d"].max()) + 0.5
    ax1.plot([0, max_val], [0, max_val], "r--", linewidth=2, label="Perfect Agreement (y = x)")
    ax1.set_title("A. Law of Cosines vs Full 3D Euclidean Distance", fontweight="bold")
    ax1.set_xlabel("Law of Cosines Distance (Metric Units)", fontweight="bold")
    ax1.set_ylabel("3D Cartesian Euclidean Distance (Metric Units)", fontweight="bold")
    ax1.legend(frameon=True)

    # Panel 2: Distance Distribution by Pair Type
    ax2 = axes[1]
    top_pairs = dist_df["pair_type"].value_counts().head(5).index.tolist()
    filtered = dist_df[dist_df["pair_type"].isin(top_pairs)]
    palette = sns.color_palette("Set2", len(top_pairs))
    sns.boxplot(data=filtered, x="pair_type", y="distance_3d", palette=palette, ax=ax2, boxprops=dict(alpha=0.8))
    sns.stripplot(data=filtered, x="pair_type", y="distance_3d", color="black", alpha=0.4, size=4, ax=ax2, jitter=0.2)
    ax2.set_title("B. 3D Distance Distribution for Top Object Pairs", fontweight="bold")
    ax2.set_xlabel("Interacting Object Pair", fontweight="bold")
    ax2.set_ylabel("Estimated 3D Distance", fontweight="bold")
    ax2.tick_params(axis="x", rotation=20)

    plt.tight_layout()
    os.makedirs(METRICS_DIR, exist_ok=True)
    plt.savefig(OUTPUT_PLOT, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved distance estimation analysis plot: {OUTPUT_PLOT}")


def main():
    print("=" * 60)
    print("DAY 4: 3D SPATIAL DISTANCE ESTIMATION (ALGORITHM 1)")
    print("=" * 60)

    dist_df = analyze_inter_object_distances()
    render_distance_annotated_frame(dist_df)
    plot_distance_analysis(dist_df)

    print("=" * 60)
    print("DAY 4 DISTANCE ESTIMATION COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()
