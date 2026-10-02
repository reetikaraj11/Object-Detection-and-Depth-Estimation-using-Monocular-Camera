"""
YOLOv8 + Depth Anything V2 — Unified Object Detection & Depth Estimation Pipeline
===================================================================================
Based on System 2 from the AI4DM course (IISc I3D Lab).

Pipeline:
  1. YOLOv8 detects objects in each video frame  (bounding boxes + class + confidence)
  2. Depth Anything V2 generates a dense depth map for the same frame
  3. Median depth within each bounding box is computed
  4. Annotated video + CSV log are saved

References:
  - Professor's reference implementation (pages 6-9 of course PDF)
  - https://cambum.net/I3DLab/AI4DM.htm
"""

import os
import csv
from tqdm import tqdm
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from transformers import pipeline

# ================== CONFIG ==================
VIDEO_IN_PATH = "data/sample_traffic.mp4"          # input video
VIDEO_OUT_PATH = "results/output_with_depth.mp4"   # output annotated video
CSV_OUT_PATH = "results/detections_with_depth.csv"  # csv output
YOLO_WEIGHTS = "weights/yolov8n.pt"                # or path to custom weights
DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"  # HF model id
IMG_SIZE_YOLO = 640                                # YOLO inference size (for speed)
DEVICE = 0                                         # 0 for GPU, "cpu" for CPU
BOX_WINDOW = 5                                     # central window size (px) for median depth
CONF_THRESH = 0.25                                 # YOLO confidence threshold
SKIP_EVERY_N = 1                                   # process every Nth frame (1 = all frames)
# =============================================


def bbox_median_depth(depth_img, x1, y1, x2, y2, window=5):
    """
    Extract median depth from a small central window within the bounding box.
    depth_img is HxW numpy array.
    """
    h, w = depth_img.shape
    x1 = max(0, int(round(x1)))
    y1 = max(0, int(round(y1)))
    x2 = min(w - 1, int(round(x2)))
    y2 = min(h - 1, int(round(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    # central small window for robustness
    if window > 1:
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        half = window // 2
        wx1, wy1 = max(0, cx - half), max(0, cy - half)
        wx2, wy2 = min(w - 1, cx + half), min(h - 1, cy + half)
        central = depth_img[wy1:wy2+1, wx1:wx2+1]
        if central.size > 0:
            return float(np.median(central))
    # fallback to full bbox median
    roi = depth_img[y1:y2+1, x1:x2+1]
    if roi.size == 0:
        return None
    return float(np.median(roi))


def main():
    # --- initialize models ---
    print("Loading YOLO model...")
    yolo = YOLO(YOLO_WEIGHTS)  # loads to default device (CPU/GPU depending on ultralytics config)
    # ensure YOLO uses same device as DEPTH pipeline where possible (ultralytics auto handles GPU)

    print("Loading Depth Anything V2 (Hugging Face) pipeline...")
    depth_pipe = pipeline(task="depth-estimation", model=DEPTH_MODEL_ID, device=DEVICE)

    # --- open video ---
    cap = cv2.VideoCapture(VIDEO_IN_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {VIDEO_IN_PATH}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(f"Video opened: {VIDEO_IN_PATH} ({width}x{height}) @ {fps} FPS, {total_frames} frames")

    # --- prepare video writer ---
    os.makedirs(os.path.dirname(VIDEO_OUT_PATH), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_vid = cv2.VideoWriter(VIDEO_OUT_PATH, fourcc, fps / max(1, SKIP_EVERY_N), (width, height))

    # --- prepare CSV ---
    os.makedirs(os.path.dirname(CSV_OUT_PATH), exist_ok=True)
    csv_file = open(CSV_OUT_PATH, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame_idx", "det_idx", "class_id", "conf", "x1", "y1", "x2", "y2", "median_depth"])

    frame_idx = 0
    processed_frames = 0
    pbar = tqdm(total=total_frames, desc="Frames")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            pbar.update(1)

            # optionally skip frames for speed
            if frame_idx % SKIP_EVERY_N != 0:
                frame_idx += 1
                continue

            # convert frame for models
            # YOLO (Ultralytics) accepts numpy BGR frames directly
            yolo_input = frame  # BGR uint8 HxWx3

            # --- run YOLO inference (single frame) ---
            # ultralytics supports direct call model(frame); set imgsz parameter in predict for resizing
            results = yolo.predict(source=yolo_input, imgsz=IMG_SIZE_YOLO, conf=CONF_THRESH, verbose=False)
            # results is list-like; for single frame, use results[0]
            res = results[0]

            # boxes in xyxy at original image scale (ultralytics scales back by default)
            if hasattr(res.boxes, "xyxy"):
                boxes = res.boxes.xyxy.cpu().numpy()  # shape (N,4)
                scores = res.boxes.conf.cpu().numpy() if hasattr(res.boxes, "conf") else [None]*len(boxes)
                classes = res.boxes.cls.cpu().numpy() if hasattr(res.boxes, "cls") else [None]*len(boxes)
            else:
                boxes = np.array([])
                scores = []
                classes = []

            # --- run depth pipeline on full frame ---
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            depth_out = depth_pipe(pil_img)
            # HF depth-estimation pipeline returns a dict with "predicted_depth" (tensor) and/or "depth" (PIL)
            if isinstance(depth_out, dict):
                if "predicted_depth" in depth_out:
                    depth_map = depth_out["predicted_depth"]
                    # may be a torch tensor — convert to numpy
                    if hasattr(depth_map, "cpu"):
                        depth_map = depth_map.squeeze().cpu().numpy()
                elif "depth" in depth_out:
                    depth_map = np.asarray(depth_out["depth"])
                else:
                    raise RuntimeError(f"Unexpected depth pipeline output keys: {depth_out.keys()}")
            elif isinstance(depth_out, list):
                depth_map = np.asarray(depth_out[0].get("depth", depth_out[0]))
            else:
                depth_map = np.asarray(depth_out)
            depth_map = depth_map.astype(np.float32)
            if depth_map.ndim == 3 and depth_map.shape[2] == 1:
                depth_map = depth_map[:, :, 0]

            # resize depth map to original resolution if needed
            h_d, w_d = depth_map.shape
            if (w_d, h_d) != (width, height):
                depth_map_resized = cv2.resize(depth_map, (width, height), interpolation=cv2.INTER_LINEAR)
            else:
                depth_map_resized = depth_map

            # --- compute depth per bbox & draw ---
            out_frame = frame.copy()
            det_idx = 0
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                conf = float(scores[i]) if len(scores) > i else None
                cls_id = int(classes[i]) if len(classes) > i else None
                med_depth = bbox_median_depth(depth_map_resized, x1, y1, x2, y2, window=BOX_WINDOW)

                # Draw bbox and text
                color = (0, 255, 0)
                cv2.rectangle(out_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                txt = f"cls:{cls_id} conf:{conf:.2f} d:{med_depth:.3f}" if med_depth is not None else f"cls:{cls_id} conf:{conf:.2f} d:N/A"
                # put text background
                (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(out_frame, (int(x1), int(y1)-th-6), (int(x1)+tw, int(y1)), color, -1)
                cv2.putText(out_frame, txt, (int(x1), int(y1)-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

                # write csv row
                csv_writer.writerow([frame_idx, det_idx, cls_id, conf, float(x1), float(y1), float(x2), float(y2), med_depth])
                det_idx += 1

            # optionally overlay a colorized depth (scaled) to visualize
            # normalize depth map to 0-255 for visualization
            vis_depth = depth_map_resized.copy().astype(np.float32)
            vis_depth = (vis_depth - vis_depth.min()) / (vis_depth.max() - vis_depth.min() + 1e-8)
            vis_depth = (vis_depth * 255).astype(np.uint8)
            vis_depth_color = cv2.applyColorMap(vis_depth, cv2.COLORMAP_JET)
            # blend: 30% depth map on top-left corner
            h_vis, w_vis = vis_depth_color.shape[:2]
            # Resize visualization to be thumbnail if original large
            thumb_w = min(320, width // 3)
            thumb_h = int(thumb_w * (h_vis / w_vis))
            vis_thumb = cv2.resize(vis_depth_color, (thumb_w, thumb_h))
            out_frame[0:thumb_h, 0:thumb_w] = cv2.addWeighted(out_frame[0:thumb_h, 0:thumb_w], 0.7, vis_thumb, 0.3, 0)

            # write frame to output
            out_vid.write(out_frame)
            processed_frames += 1
            frame_idx += 1

    finally:
        pbar.close()
        cap.release()
        out_vid.release()
        csv_file.close()

    print(f"Done. Processed frames: {processed_frames}. Output video: {VIDEO_OUT_PATH}, CSV: {CSV_OUT_PATH}")


if __name__ == "__main__":
    main()
