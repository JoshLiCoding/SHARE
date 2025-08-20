import os
import numpy as np
import argparse
import xml.etree.ElementTree as ET

def load_cam_number_from_name(name):
    import re
    m = re.search(r'_cam(\d+)_', name)
    if m:
        return int(m.group(1))
    return None

def load_calib_matrix(calib_folder, scan_name, cam_num):
    # Example path: calib_folder/scan_name/calibration/002.xml
    cam_file = f"{cam_num:03d}.xml"
    calib_path = os.path.join(calib_folder, scan_name, "calibration", cam_file)
    # Parse XML
    tree = ET.parse(calib_path)
    root = tree.getroot()
    cam_matrix_elem = root.find("CameraMatrix")
    cam_data = cam_matrix_elem.find('data').text.strip().split()
    cam_matrix = np.array([float(x) for x in cam_data]).reshape(3, 4)
    R = cam_matrix[:, :3]
    t = cam_matrix[:, 3]
    return R, t

def inverse_transform(pts, R, t):
        R_inv = np.linalg.inv(R)
        return (pts - t) @ R_inv.T

def main():
    parser = argparse.ArgumentParser(description="Evaluate joint/vertex errors for matching subfolders")
    parser.add_argument("gt_folder", type=str, help="Path to first large folder (ground truth)")
    parser.add_argument("pred_folder", type=str, help="Path to second large folder (predictions)")
    parser.add_argument("--pred_sub2folder", type=str, default=None, help="Optional: specific subfolder to evaluate")
    parser.add_argument("--joints_name", type=str, default="joints.npy", help="Filename for joints in pred_folder")
    parser.add_argument("--verts_name", type=str, default="vertices.npy", help="Filename for vertices in pred_folder")
    parser.add_argument("--calib_folder", type=str, default="../rich_toolkit/data/scan_calibration", help="Path to calibration files root")
    args = parser.parse_args()

    def strip_camx(name):
        import re
        return re.sub(r'_cam\d+_', '_', name)

    subfolders_gt = set(os.listdir(args.gt_folder))
    subfolders_pred = set(os.listdir(args.pred_folder))
    # Map: stripped2name -> original2name
    pred_map = {strip_camx(s2): s2 for s2 in subfolders_pred}
    common_subfolders = sorted([s1 for s1 in subfolders_gt if s1 in pred_map])
    if len(common_subfolders) != len(subfolders_gt):
        print("Warning: Not all subfolders match between the two folders after removing _camx_.")
        print(f"GT subfolders: {len(subfolders_gt)}, Pred subfolders: {len(subfolders_pred)}")

    # Collect errors for all frames across all sequences
    all_mrpe_errors = []
    all_v2v_errors = []

    for sub in common_subfolders:
        path_gt = os.path.join(args.gt_folder, sub)
        sub_pred = pred_map[sub]
        path_pred = os.path.join(args.pred_folder, sub_pred)

        # Load data for gt (default names)
        joints_gt = np.load(os.path.join(path_gt, "joints.npy"))
        verts_gt = np.load(os.path.join(path_gt, "vertices.npy"))
        # Load data for pred (custom names)
        if args.pred_sub2folder:
            path_pred = os.path.join(path_pred, args.pred_sub2folder)
        joints_pred_path = os.path.join(path_pred, args.joints_name)
        verts_pred_path = os.path.join(path_pred, args.verts_name)
        if not os.path.exists(joints_pred_path) or not os.path.exists(verts_pred_path):
            print(f"Warning: {joints_pred_path} or {verts_pred_path} does not exist for {sub}, skipping.")
            continue
        joints_pred = np.load(joints_pred_path)
        verts_pred = np.load(verts_pred_path)
        if joints_gt.shape != joints_pred.shape or verts_gt.shape != verts_pred.shape:
            print(f"Warning: Shape mismatch for {sub}: "
                  f"GT joints {joints_gt.shape}, Pred joints {joints_pred.shape}, "
                  f"GT verts {verts_gt.shape}, Pred verts {verts_pred.shape}. Skipping")
            continue

        # --- Camera transform for pred ---
        cam_num = load_cam_number_from_name(sub_pred)
        if cam_num is None:
            print(f"Warning: Could not extract camera number from {sub_pred}, skipping.")
            continue
        scan_name = sub_pred.split('_')[0]
        R, t = load_calib_matrix(args.calib_folder, scan_name, cam_num)

        # Inverse transform pred from camera to world
        # (Assume pred is in camera frame, GT is in world frame)
        joints_pred_world = inverse_transform(joints_pred, R, t)
        verts_pred_world = inverse_transform(verts_pred, R, t)

        # Calculate errors for all frames in the sequence
        frame_mrpe = np.linalg.norm(joints_gt[:, 0, :] - joints_pred_world[:, 0, :], axis=1)  # MRPE per frame
        frame_v2v = np.linalg.norm(verts_gt - verts_pred_world, axis=2).mean(axis=1)  # V2V per frame

        # Append frame-level errors to the global list
        all_mrpe_errors.extend(frame_mrpe)
        all_v2v_errors.extend(frame_v2v)

        # Sequence-level averages (for logging)
        mrpe = np.mean(frame_mrpe)
        v2v = np.mean(frame_v2v)
        print(f"{sub}: MRPE={mrpe:.4f}, V2V={v2v:.4f}")

    # Calculate global statistics across all frames
    avg_mrpe = np.mean(all_mrpe_errors) if all_mrpe_errors else float('nan')
    std_mrpe = np.std(all_mrpe_errors, ddof=1) if all_mrpe_errors else float('nan')  # Standard deviation for MRPE
    avg_v2v = np.mean(all_v2v_errors) if all_v2v_errors else float('nan')
    std_v2v = np.std(all_v2v_errors, ddof=1) if all_v2v_errors else float('nan')  # Standard deviation for V2V

    print(f"\nAverage Mean Root Position Error (MRPE): {avg_mrpe:.2f}")
    print(f"Standard Deviation of MRPE: {std_mrpe:.2f}")
    print(f"Average Vertex-to-Vertex Error (V2V): {avg_v2v:.2f}")
    print(f"Standard Deviation of V2V: {std_v2v:.2f}")

if __name__ == "__main__":
    main()