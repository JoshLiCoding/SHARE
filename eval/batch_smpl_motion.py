import os
import pickle
import numpy as np

input_root = 'eval/RICH/smpl_motions'
output_root = 'eval/RICH/batched_smpl_motions'

for folder in sorted(os.listdir(input_root)):
    folder_path = os.path.join(input_root, folder)
    if not os.path.isdir(folder_path):
        continue
    # Numeric sort for .pkl files
    def numeric_key(fname):
        base = os.path.splitext(fname)[0]
        try:
            return int(base)
        except ValueError:
            return base
    pkl_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.pkl')], key=numeric_key)
    if len(pkl_files) != 100:
        print(f"Warning: {folder} does not have 100 .pkl files, skipping.")
        continue
    joints_list = []
    verts_list = []
    for pkl_file in pkl_files:
        with open(os.path.join(folder_path, pkl_file), 'rb') as f:
            data = pickle.load(f)
        joints = data['joints'].detach().cpu().numpy()[0, :24]  # (24, 3)
        verts = data['vertices'].detach().cpu().numpy()[0]      # (6890, 3)
        joints_list.append(joints)
        verts_list.append(verts)
    joints_arr = np.stack(joints_list, axis=0)  # (100, 24, 3)
    verts_arr = np.stack(verts_list, axis=0)    # (100, 6890, 3)
    # Prepare output directory
    out_folder = os.path.join(output_root, folder)
    os.makedirs(out_folder, exist_ok=True)
    out_joints = os.path.join(out_folder, 'joints.npy')
    out_verts = os.path.join(out_folder, 'vertices.npy')
    np.save(out_joints, joints_arr)
    np.save(out_verts, verts_arr)
    print(f"Processed {folder}: saved to {out_folder}")
