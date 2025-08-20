import os
import subprocess

TRAM_RESULTS_DIR = 'eval/Toyota/share'
INFER_SCRIPT = '../MoGe/moge/scripts/infer.py'

for folder in os.listdir(TRAM_RESULTS_DIR):
    folder_path = os.path.join(TRAM_RESULTS_DIR, folder)
    if not os.path.isdir(folder_path):
        continue

    images_dir = os.path.join(folder_path, 'images')
    if not os.path.exists(images_dir) or not os.path.isdir(images_dir):
        print(f"Images folder not found: {images_dir}")
        continue

    # Get all image files in the folder, sorted by name
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))])
    if len(image_files) < 2:
        print(f"Not enough images in {images_dir} to process.")
        continue

    # Select the 1st and last image
    first_image = os.path.join(images_dir, image_files[0])
    last_image = os.path.join(images_dir, image_files[-1])

    # Process the selected images
    for image_path in [first_image, last_image]:
        output_dir = folder_path
        cmd = [
            'python', INFER_SCRIPT,
            '-i', image_path,
            '-o', output_dir,
            '--maps', '--ply'
        ]
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)