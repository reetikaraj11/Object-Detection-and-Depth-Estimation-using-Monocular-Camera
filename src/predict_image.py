"""
Single Image / Static Frame Prediction Utility
================================================
Runs the unified YOLOv8 + Depth Anything V2 pipeline on a single image or video frame:
  1. Object Detection (YOLOv8n) -> Bounding boxes, classes, confidences
  2. Dense Depth Estimation (Depth Anything V2 Small) -> Aligned depth field
  3. Robust Centroid Sampling (5x5 window) -> Median depth per object
  4. 3D Spatial Geometry (Algorithm 1) -> Inter-object distances
  5. Renders annotated visualization with thumbnail depth map and distance lines

Usage:
  python src/predict_image.py [--image <path_to_image>] [--conf <threshold>] [--out <output_path>]

Author: Richi (I3D Lab, Indian Institute of Science, Bangalore)
Course: AI for Design & Manufacturing (AI4DM)
"""

import os
import sys
import argparse
import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
from transformers import pipeline

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_WEIGHTS = os.path.join(PROJECT_ROOT, "weights", "yolov8n.pt")
DEFAULT_OUT = os.path.join(PROJECT_ROOT, "results", "sample_frames", "single_image_prediction.png")
DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"


def bbox_median_depth(depth_map, x1, y1, x2, y2, window=5):
    """Extract median depth within a central window inside the bounding box."""
    h, w = depth_map.shape[:2]
    x1 = max(0, int(round(x1)))
    y1 = max(0, int(round(y1)))
    x2 = min(w - 1, int(round(x2)))
    y2 = min(h - 1, int(round(y2)))

    if x2 <= x1 or y2 <= y1:
        return None

    if window > 1:
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        half = window // 2
        wx1, wy1 = max(0, cx - half), max(0, cy - half)
        wx2, wy2 = min(w - 1, cx + half), min(h - 1, cy + half)
        central = depth_map[wy1:wy2+1, wx1:wx2+1]
        if central.size > 0:
            return float(np.median(central))

    roi = depth_map[y1:y2+1, x1:x2+1]
    return float(np.median(roi)) if roi.size > 0 else None


def predict_single_image(image_path=None, conf_thresh=0.25, output_path=DEFAULT_OUT):
    print("=" * 70)
    print("YOLOv8 + DEPTH ANYTHING V2: STATIC IMAGE INFERENCE")
    print("=" * 70)

    # 1. Load or extract test image
    if image_path is None or not os.path.exists(image_path):
        video_path = os.path.join(PROJECT_ROOT, "data", "sample_traffic.mp4")
        if os.path.exists(video_path):
            print(f"Extracting sample frame #324 from {video_path}...")
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 324)
            ret, frame = cap.read()
            cap.release()
            if not ret or frame is None:
                raise RuntimeError("Failed to extract frame from video.")
        else:
            raise FileNotFoundError(f"Input image not found: {image_path}")
    else:
        print(f"Loading input image: {image_path}")
        frame = cv2.imread(image_path)
        if frame is None:
            raise RuntimeError(f"Could not read image: {image_path}")

    h, w = frame.shape[:2]
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Image Resolution: {w}x{h} | Compute Device: {device}")

    # 2. Load Models
    print("Loading YOLOv8n detector...")
    yolo_model = YOLO(DEFAULT_WEIGHTS)

    print("Loading Depth Anything V2 Small pipeline...")
    depth_pipe = pipeline(task="depth-estimation", model=DEPTH_MODEL_ID, device=device)

    # 3. YOLO Detection
    print(f"Running detection (confidence threshold: {conf_thresh})...")
    results = yolo_model.predict(source=frame, imgsz=640, conf=conf_thresh, verbose=False)[0]

    boxes = results.boxes.xyxy.cpu().numpy() if hasattr(results.boxes, "xyxy") else np.array([])
    scores = results.boxes.conf.cpu().numpy() if hasattr(results.boxes, "conf") else []
    classes = results.boxes.cls.cpu().numpy() if hasattr(results.boxes, "cls") else []
    names = results.names

    print(f"Detected {len(boxes)} object(s).")

    # 4. Depth Estimation
    print("Running Depth Anything V2 dense estimation...")
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    depth_out = depth_pipe(pil_img)
    if isinstance(depth_out, dict):
        if "predicted_depth" in depth_out:
            depth_map = depth_out["predicted_depth"]
            if hasattr(depth_map, "cpu"):
                depth_map = depth_map.squeeze().cpu().numpy()
            else:
                depth_map = np.asarray(depth_map)
        elif "depth" in depth_out:
            depth_map = np.asarray(depth_out["depth"])
        else:
            depth_map = np.asarray(list(depth_out.values())[0])
    elif isinstance(depth_out, list):
        first = depth_out[0]
        if isinstance(first, dict) and "depth" in first:
            depth_map = np.asarray(first["depth"])
        else:
            depth_map = np.asarray(first)
    else:
        depth_map = np.asarray(depth_out)

    if depth_map.ndim == 3 and depth_map.shape[2] == 1:
        depth_map = depth_map[:, :, 0]

    if depth_map.shape[:2] != (h, w):
        depth_map = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_LINEAR)

    # 5. Extract Depths & 3D Coordinates
    fx = fy = 0.8 * w
    cx0 = w / 2.0
    cy0 = h / 2.0

    detections = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        conf = float(scores[i])
        cls_id = int(classes[i])
        cls_name = names.get(cls_id, str(cls_id))
        d = bbox_median_depth(depth_map, x1, y1, x2, y2, window=5)

        if d is not None:
            cx_pix = (x1 + x2) / 2.0
            cy_pix = (y1 + y2) / 2.0
            X = (cx_pix - cx0) * d / fx
            Y = (cy_pix - cy0) * d / fy
            Z = d
            detections.append({
                "idx": i, "box": (x1, y1, x2, y2), "conf": conf,
                "class": cls_name, "depth": d, "centroid": (cx_pix, cy_pix),
                "coord_3d": (X, Y, Z)
            })

    # 6. Render Output Frame
    out_frame = frame.copy()

    # Color palette
    colors_list = [(46, 204, 113), (52, 152, 219), (155, 89, 182), (241, 196, 15), (230, 126, 34)]

    # Draw pairwise distance lines between nearby objects
    for i in range(len(detections)):
        for j in range(i + 1, len(detections)):
            d1 = detections[i]
            d2 = detections[j]
            p1 = (int(d1["centroid"][0]), int(d1["centroid"][1]))
            p2 = (int(d2["centroid"][0]), int(d2["centroid"][1]))
            dist_3d = np.linalg.norm(np.array(d1["coord_3d"]) - np.array(d2["coord_3d"]))

            # Draw dotted/dashed line
            cv2.line(out_frame, p1, p2, (255, 255, 0), 1, cv2.LINE_AA)
            mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
            cv2.putText(out_frame, f"{dist_3d:.2f}m", (mid[0]-15, mid[1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)

    # Draw bounding boxes and depth tags
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        col = colors_list[det["idx"] % len(colors_list)]
        cv2.rectangle(out_frame, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
        
        tag = f"{det['class']} {det['conf']:.2f} | d={det['depth']:.2f}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(out_frame, (int(x1), int(y1) - th - 6), (int(x1) + tw + 6, int(y1)), col, -1)
        cv2.putText(out_frame, tag, (int(x1) + 3, int(y1) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    # Overlay colorized depth map thumbnail (top-left)
    norm_depth = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
    vis_depth = cv2.applyColorMap((norm_depth * 255).astype(np.uint8), cv2.COLORMAP_JET)
    thumb_w = min(240, w // 3)
    thumb_h = int(thumb_w * (h / w))
    vis_thumb = cv2.resize(vis_depth, (thumb_w, thumb_h))
    out_frame[8:8+thumb_h, 8:8+thumb_w] = cv2.addWeighted(
        out_frame[8:8+thumb_h, 8:8+thumb_w], 0.25, vis_thumb, 0.75, 0
    )
    cv2.rectangle(out_frame, (8, 8), (8 + thumb_w, 8 + thumb_h), (255, 255, 255), 1)
    cv2.putText(out_frame, "Depth Map", (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Save output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, out_frame)
    print(f"\nResult saved to: {output_path}")

    # Summary table
    print("\n" + "-" * 70)
    print(f"{'ID':<4} {'Class':<12} {'Conf':<8} {'Depth':<8} {'3D Coord (X, Y, Z)'}")
    print("-" * 70)
    for det in detections:
        c3 = det["coord_3d"]
        print(f"{det['idx']:<4} {det['class']:<12} {det['conf']:<8.2f} {det['depth']:<8.2f} ({c3[0]:.2f}, {c3[1]:.2f}, {c3[2]:.2f})")
    print("=" * 70)
    return output_path, len(detections)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict single image with YOLO + Depth Anything V2")
    parser.add_argument("--image", default=None, help="Path to input image")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Path to output annotated image")
    args = parser.parse_args()

    predict_single_image(args.image, args.conf, args.out)
