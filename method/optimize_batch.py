import os
import contextlib
import json
import numpy as np
import torch
import torch.optim as optim
import cv2
from glob import glob
from smplx import SMPL
import trimesh
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
zero = torch.tensor(0, device=device)
one = torch.tensor(1, device=device)

def load(folder, frame):
    image = cv2.imread(os.path.join(folder, frame, 'image.jpg'))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    depth = cv2.imread(os.path.join(folder, frame, 'depth.exr'), cv2.IMREAD_UNCHANGED)
    human_mask = cv2.imread(os.path.join(folder, 'Annotations', f'{frame}.png'))
    human_mask = (human_mask.sum(axis=2) > 0).astype(bool)
    mask_cleaned = cv2.imread(os.path.join(folder, frame, 'mask_cleaned.png'), cv2.IMREAD_GRAYSCALE) > 0
    with open(os.path.join(folder, frame, 'fov.json')) as f:
        fov = json.load(f)
    return image, depth, human_mask, mask_cleaned, fov

def load_smpl_param(folder):
    hps_folder = f'{folder}/hps'
    hps_files = glob(f'{hps_folder}/*.npy')
    hps_file = hps_files[0] # Assuming 1 human in frame

    pred_smpl = np.load(hps_file, allow_pickle=True).item()
    pred_rotmat = pred_smpl['pred_rotmat'].to(device)
    pred_shape = pred_smpl['pred_shape'].to(device)
    pred_trans = pred_smpl['pred_trans'].to(device)

    mean_shape = pred_shape.mean(dim=0, keepdim=True)

    return pred_rotmat, mean_shape, pred_trans

def traj_filter(pred_j3d, pred_vert, sigma=3):
    """ Smooth the root trajectory (xyz) using a Gaussian kernel with PyTorch """
    root = pred_j3d[:, 0]  # [T, 3]
    T, D = root.shape

    # Create Gaussian kernel
    kernel_size = int(6 * sigma + 1)
    x = torch.arange(kernel_size) - kernel_size // 2
    kernel = torch.exp(-0.5 * (x.float() / sigma) ** 2)
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, -1).to(device)

    # Pad and filter each dimension with reflect mode
    pad = kernel_size // 2
    root_t = root.T.unsqueeze(0)  # [1, 3, T]
    root_t_padded = torch.nn.functional.pad(root_t, (pad, pad), mode='reflect')
    root_smooth = torch.nn.functional.conv1d(
        root_t_padded, kernel.expand(3, 1, kernel_size), padding=0, groups=3
    ).squeeze(0).T  # [T, 3]

    pred_vert = pred_vert + (root_smooth - root)[:, None]
    pred_j3d = pred_j3d + (root_smooth - root)[:, None]
    return pred_j3d, pred_vert

def create_smpl_body(pred_rotmat, mean_shape, pred_trans, smooth_traj=False):
    with contextlib.redirect_stdout(None):
        smpl = SMPL(model_path='body_models/smpl').to(device)
    
    pred_shape = mean_shape.repeat(pred_rotmat.shape[0], 1)
    pred = smpl(body_pose=pred_rotmat[:, 1:],
                global_orient=pred_rotmat[:, [0]],
                betas=pred_shape,
                transl=pred_trans.squeeze(1),
                pose2rot=False)
    pred_vert = pred.vertices
    pred_faces = smpl.faces
    pred_j3d = pred.joints[:, :24]  # Use only the first 24 joints
    if smooth_traj:
        pred_j3d, pred_vert = traj_filter(pred_j3d, pred_vert)
    return pred_j3d, pred_vert

def save_ply(filename, points, colors):
    colors = colors.astype('uint8')
    cloud = trimesh.PointCloud(vertices=points, colors=colors)
    cloud.export(filename, file_type='ply', encoding='binary')

def project_to_3d(depth, mask, fov, scale=1.0):
    fov_x = torch.deg2rad(torch.tensor(fov['fov_x'], dtype=torch.float32, device=device))
    fov_y = torch.deg2rad(torch.tensor(fov['fov_y'], dtype=torch.float32, device=device))
    H, W = depth.shape
    cx = W / 2
    cy = H / 2
    fx = W / (2 * torch.tan(fov_x / 2))
    fy = H / (2 * torch.tan(fov_y / 2))

    ys, xs = np.where(mask)
    zs = torch.tensor(depth[ys, xs], dtype=torch.float32, device=device) * scale
    xs = torch.tensor(xs, dtype=torch.float32, device=device)
    ys = torch.tensor(ys, dtype=torch.float32, device=device)

    X = (xs - cx) * zs / fx
    Y = (ys - cy) * zs / fy
    Z = zs
    points = torch.stack([X, Y, Z], dim=1)
    return points

def random_subset(pc, num_points=5000):
    if pc.shape[0] > num_points:
        idx = torch.randperm(pc.shape[0])[:num_points]
        return pc[idx]
    return pc

def chamfer_distance(pc1, pc2):
    pc1 = random_subset(pc1)
    pc2 = random_subset(pc2)
    diff = torch.cdist(pc1, pc2)  # [N, M]
    min1 = diff.min(dim=0)[0]     # [M]
    min2 = diff.min(dim=1)[0]     # [N]
    return min1.mean() + min2.mean()

def main(folder, frame_1, frame_2):
    # Load data
    image1, depth1, human_mask1, mask_cleaned1, fov1 = load(folder, f"{frame_1:04d}")
    image2, depth2, human_mask2, mask_cleaned2, fov2 = load(folder, f"{frame_2:04d}")

    common_background_mask = ~human_mask1 & ~human_mask2 & mask_cleaned1 & mask_cleaned2
    d1 = depth1[common_background_mask]
    d2 = depth2[common_background_mask]
    scale2 = np.mean(d1) / np.mean(d2)

    pred_rotmat, mean_shape, pred_trans = load_smpl_param(folder)
    refined_trans = torch.nn.Parameter(pred_trans.clone().detach())

    init_root_trans = create_smpl_body(pred_rotmat, mean_shape, pred_trans, smooth_traj=True)[0][:, 0, :]
    init_rel_root_trans1 = init_root_trans - init_root_trans[frame_1]
    init_rel_root_trans2 = init_root_trans - init_root_trans[frame_2]

    n_iters=600
    lr=1e-2
    
    optimizer = optim.Adam([refined_trans], lr=lr)
    for it in range(n_iters): 
        optimizer.zero_grad()

        # Loss 1: Keyframe body matching
        human_pc1 = project_to_3d(depth1, human_mask1 & mask_cleaned1, fov1)
        human_pc2 = project_to_3d(depth2, human_mask2 & mask_cleaned2, fov2, scale2)

        pred_vert = create_smpl_body(pred_rotmat, mean_shape, refined_trans)[1]
        keyframe_loss1 = chamfer_distance(pred_vert[frame_1], human_pc1)
        keyframe_loss2 = chamfer_distance(pred_vert[frame_2], human_pc2)
        keyframe_loss = keyframe_loss1 + keyframe_loss2
        
        # Loss 2: Relative root translation preservation (wrt to keyframes)
        root_trans = create_smpl_body(pred_rotmat, mean_shape, refined_trans)[0][:, 0, :]
        rel_root_trans1 = root_trans - root_trans[frame_1]
        rel_root_trans2 = root_trans - root_trans[frame_2]

        mask_exclude_frame2 = torch.ones(len(refined_trans), dtype=torch.bool)
        mask_exclude_frame2[frame_2] = False
        mask_exclude_frame1 = torch.ones(len(refined_trans), dtype=torch.bool)
        mask_exclude_frame1[frame_1] = False
        
        rel_root_trans_loss = torch.nn.functional.mse_loss(rel_root_trans1[mask_exclude_frame2], init_rel_root_trans1[mask_exclude_frame2]) + \
                              torch.nn.functional.mse_loss(rel_root_trans2[mask_exclude_frame1], init_rel_root_trans2[mask_exclude_frame1])
        
        loss = keyframe_loss + rel_root_trans_loss
        loss.backward()
        optimizer.step()

        if it % 20 == 0 or it == (n_iters // 3) - 1:
            print(f"Iteration {it}, Loss: {loss.item():.3f}, Keyframe Loss: {keyframe_loss.item():.3f}, Relative Root Trans Loss: {rel_root_trans_loss.item():.3f}")
    
    print("Optimization finished.")
    
    pc1_background = project_to_3d(depth1, common_background_mask, fov1)
    pc2_background = project_to_3d(depth2, common_background_mask, fov2, scale2)
    pc_background_mean = (pc1_background + pc2_background) / 2
    background_colors = image1[common_background_mask].reshape(-1, 3)

    # Take human mask from the other point cloud
    pc1_human = project_to_3d(depth2, human_mask1 & mask_cleaned2, fov2, scale2)
    pc2_human = project_to_3d(depth1, human_mask2 & mask_cleaned1, fov1)
    pc1_human_colors = image2[human_mask1 & mask_cleaned2].reshape(-1, 3)
    pc2_human_colors = image1[human_mask2 & mask_cleaned1].reshape(-1, 3)
    
    pc_scene = torch.cat([pc_background_mean, pc1_human, pc2_human], dim=0)
    colors_scene = np.concatenate([background_colors, pc1_human_colors, pc2_human_colors], axis=0)
    save_ply(os.path.join(folder, 'share_scene.ply'), pc_scene.cpu().numpy(), colors_scene)

    # Save pred_verts.npy and pred_j3d.npy directly in the subfolder
    pred_j3d, pred_vert = create_smpl_body(pred_rotmat, mean_shape, refined_trans)
    np.save(os.path.join(folder, 'share_vertices.npy'), pred_vert.detach().cpu().numpy())
    np.save(os.path.join(folder, 'share_joints.npy'), pred_j3d.detach().cpu().numpy())

if __name__ == "__main__":
    import argparse
    TRAM_RESULTS = "eval/RICH/tram_share_results"
    FRAME_1 = 0
    FRAME_2 = 99

    subfolders = [d for d in os.listdir(TRAM_RESULTS) if os.path.isdir(os.path.join(TRAM_RESULTS, d))]
    for sub in sorted(subfolders):
        folder = os.path.join(TRAM_RESULTS, sub)
        try:
            print(f"Optimizing {sub}...")
            main(folder, FRAME_1, FRAME_2)
            print(f"{sub} done.")
        except Exception as e:
            print(f"Skipping {sub}: {e}")