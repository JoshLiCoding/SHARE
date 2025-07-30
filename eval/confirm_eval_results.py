import os
import re

def strip_camx(name):
    """Remove _camX_ from a folder name."""
    return re.sub(r'_cam\d+_', '_', name)

def get_stripped_subfolders(folder):
    """Return a set of subfolder names with _camX_ removed."""
    return set(strip_camx(name) for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name)))

def main():
    folders = [
        "eval/RICH/batched_smpl_motions",
        "eval/RICH/bev_results",
        "eval/RICH/mhm_results",
        "eval/RICH/tram_share_results"
    ]

    sets = [get_stripped_subfolders(f) for f in folders]

    # Check if all sets are equal
    all_equal = all(s == sets[0] for s in sets[1:])
    same_length = all(len(s) == len(sets[0]) for s in sets)

    if all_equal and same_length:
        print("All folders have the same subfolder names (ignoring _camx_) and the same number of subfolders.")
    else:
        print("Mismatch found!")
        for i, s in enumerate(sets):
            print(f"{folders[i]}: {len(s)} unique subfolders")
        # Show differences
        for i in range(len(sets)):
            for j in range(i+1, len(sets)):
                diff = sets[i] - sets[j]
                if diff:
                    print(f"Subfolders in {folders[i]} but not in {folders[j]}: {sorted(diff)}")

if __name__ == "__main__":
    main()