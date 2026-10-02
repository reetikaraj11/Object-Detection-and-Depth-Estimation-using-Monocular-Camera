"""
Download or generate sample test video and test image for the YOLO + Depth pipeline.
"""
import os
import urllib.request
import cv2
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

def download_or_create_sample():
    os.makedirs(DATA_DIR, exist_ok=True)
    video_path = os.path.join(DATA_DIR, "sample_traffic.mp4")
    
    # Check if already exists
    if os.path.exists(video_path) and os.path.getsize(video_path) > 1000:
        print(f"[OK] Sample video already exists: {video_path}")
        return video_path

    # Try downloading a public domain / open sample traffic clip
    url = "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/person-bicycle-car-detection.mp4"
    print(f"Downloading sample video from {url}...")
    try:
        urllib.request.urlretrieve(url, video_path)
        print(f"[OK] Successfully downloaded sample video to {video_path}")
        return video_path
    except Exception as e:
        print(f"[WARN] Network download failed ({e}), generating a realistic synthetic video instead...")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        width, height, fps = 640, 480, 30
        out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        
        for i in range(150):
            frame = np.ones((height, width, 3), dtype=np.uint8) * 180  # Road background
            # Draw lane lines
            cv2.line(frame, (width // 2, 0), (width // 2, height), (255, 255, 255), 3)
            # Moving "car" 1
            x1 = int(100 + i * 2.5) % width
            cv2.rectangle(frame, (x1, 200), (x1 + 90, 260), (40, 40, 200), -1)
            cv2.putText(frame, "Car", (x1 + 10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            # Moving "person" 2
            y2 = int(50 + i * 1.5) % height
            cv2.rectangle(frame, (450, y2), (480, y2 + 60), (0, 150, 0), -1)
            cv2.putText(frame, "Person", (430, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 0), 2)
            
            out.write(frame)
        out.release()
        print(f"[OK] Generated synthetic test video at: {video_path}")
        return video_path

if __name__ == "__main__":
    download_or_create_sample()
