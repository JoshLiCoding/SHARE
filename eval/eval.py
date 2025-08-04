import os
import numpy as np
import argparse
import xml.etree.ElementTree as ET

def mean_root_position_error(joints1, joints2):
    # Root is usually joint 0
    root1 = joints1[:, 0, :]  # shape: (T, 3)
    root2 = joints2[:, 0, :]
    error = np.linalg.norm(root1 - root2, axis=1)  # shape: (T,)
    return np.mean(error)

def vertex_to_vertex_error(verts1, verts2):
    error = np.linalg.norm(verts1 - verts2, axis=2)  # shape: (T, V)
    return np.mean(error)

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

    mrpe_list = []
    v2v_list = []

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

        mrpe = mean_root_position_error(joints_gt, joints_pred_world)
        v2v = vertex_to_vertex_error(verts_gt, verts_pred_world)
        mrpe_list.append(mrpe)
        v2v_list.append(v2v)
        print(f"{sub}: MRPE={mrpe:.4f}, V2V={v2v:.4f}")

    avg_mrpe = np.mean(mrpe_list) if mrpe_list else float('nan')
    avg_v2v = np.mean(v2v_list) if v2v_list else float('nan')

    print(f"\nAverage Mean Root Position Error (MRPE): {avg_mrpe:.4f}")
    print(f"Average Vertex-to-Vertex Error (V2V): {avg_v2v:.4f}")

if __name__ == "__main__":
    main()