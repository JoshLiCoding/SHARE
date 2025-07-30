import os
import cv2

videos_dir = "eval/RICH/videos"
frames_root = "eval/RICH/frames"

for fname in os.listdir(videos_dir):
    if not fname.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        continue
    video_path = os.path.join(videos_dir, fname)
    video_name, _ = os.path.splitext(fname)
    out_dir = os.path.join(frames_root, video_name)
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_idx = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out_path = os.path.join(out_dir, f"{frame_idx:03d}.jpg")
        cv2.imwrite(out_path, frame)
        frame_idx += 1
    cap.release()
    print(f"Processed {fname}: {frame_idx-1} frames saved to {out_dir}")