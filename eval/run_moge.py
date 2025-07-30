import os
import subprocess

TRAM_RESULTS_DIR = 'eval/RICH/tram_results'
INFER_SCRIPT = '../MoGe/moge/scripts/infer.py'

for folder in os.listdir(TRAM_RESULTS_DIR):
    folder_path = os.path.join(TRAM_RESULTS_DIR, folder)
    if not os.path.isdir(folder_path):
        continue
    # Compose image paths for 0th and 99th frames
    for frame in [0, 99]:
        image_path = os.path.join(TRAM_RESULTS_DIR, folder, 'images', f'{frame:04d}.jpg')
        output_dir = os.path.join(TRAM_RESULTS_DIR, folder)
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue
        cmd = [
            'python', INFER_SCRIPT,
            '-i', image_path,
            '-o', output_dir,
            '--maps', '--ply'
        ]
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
