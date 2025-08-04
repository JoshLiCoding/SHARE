import os
import random
import json
from tqdm import tqdm
import shutil
import imageio

# Paths
BASE_DIR = "../rich_toolkit/data/bodies/test"
IMG_BASE_DIR = "../rich_toolkit/data/images/test"
OUTPUT_DIR = "eval/RICH"
OUTPUT_MOTION_DIR = os.path.join(OUTPUT_DIR, "motions")
OUTPUT_VIDEO_DIR = os.path.join(OUTPUT_DIR, "videos")

# Create output directories
os.makedirs(OUTPUT_MOTION_DIR, exist_ok=True)
os.makedirs(OUTPUT_VIDEO_DIR, exist_ok=True)

def process_sequence(seq_path, seq_name, cam_num):
    """Copy ply files for a single sequence to eval/RICH/motions/ in batches of 100 frames and save videos with a specified camera per sequence."""
    print(f"Processing sequence: {seq_name} with camera {cam_num:02d}")

    # Find all 5-digit numbered folders (frames)
    frame_folders = []
    for item in os.listdir(seq_path):
        if os.path.isdir(os.path.join(seq_path, item)) and item.isdigit() and len(item) == 5:
            frame_folders.append(item)
    
    if not frame_folders:
        print(f"No frame folders found in {seq_name}, skipping")
        return

    # Sort frame folders numerically
    frame_folders.sort(key=int)
    
    # Get subject id from seq_name
    _, sub_id, _ = seq_name.split('_')
    cam_dir = f"cam_{cam_num:02d}"

    # Process in batches of 100
    for batch_idx in range(0, len(frame_folders), 100):
        batch = frame_folders[batch_idx:batch_idx+100]
        if not batch:
            continue

        # Output directory for this batch
        batch_dir = os.path.join(OUTPUT_MOTION_DIR, f"{seq_name}_batch{batch_idx//100:02d}")
        os.makedirs(batch_dir, exist_ok=True)

        images = []
        for frame_folder in tqdm(batch, desc=f"Copying ply/img for {seq_name} batch {batch_idx//100:02d}"):
            frame_path = os.path.join(seq_path, frame_folder)
            ply_file = os.path.join(frame_path, f"{sub_id}.ply")
            unique_name = f"{frame_folder}.ply"
            out_ply = os.path.join(batch_dir, unique_name)
            if os.path.exists(ply_file):
                shutil.copy2(ply_file, out_ply)
            else:
                print(f"Missing ply: {ply_file}")
            
            img_name = f"{frame_folder}_{cam_num:02d}.jpeg"
            img_path = os.path.join(IMG_BASE_DIR, seq_name, cam_dir, img_name)
            if os.path.exists(img_path):
                images.append(imageio.v2.imread(img_path))
            else:
                print(f"Missing image: {img_path}")

        # Save video for this batch if images are available
        if images:
            video_path = os.path.join(OUTPUT_VIDEO_DIR, f"{seq_name}_cam{cam_num}_batch{batch_idx//100:02d}.mp4")
            imageio.mimsave(video_path, images, fps=30)

def main():
    # Load selected sequences and their camera numbers from JSON
    seq_cam_json_path = "eval/sequence_camera_numbers.json"
    with open(seq_cam_json_path, "r") as f:
        seq_cam_map = json.load(f)

    selected_seqs = list(seq_cam_map.keys())

    for seq_name in selected_seqs:
        seq_path = os.path.join(BASE_DIR, seq_name)
        cam_num = seq_cam_map[seq_name]
        process_sequence(seq_path, seq_name, cam_num)

    print("Processing complete!")

if __name__ == "__main__":
    main()