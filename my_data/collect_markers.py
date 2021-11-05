import os.path

import numpy as np
import pandas as pd


skip_existing = True

trial_dirs = []
with open("trials.txt", "r") as f:
    for line in f:
        if line[:1] == "#":
            continue
        trial_dirs.append(line.rstrip())

for trial_dir in trial_dirs:
    descriptor = trial_dir.replace("/mnt/data2/FA/", "").rstrip("/").replace("/", "_")
    output_file = f"markers/{descriptor}_labeled.npy"
    
    print(output_file)

    if skip_existing and os.path.isfile(output_file):
        continue

    input_file = os.path.join(trial_dir, "behData/images/df3d/post_processed.pkl")
    df = pd.read_pickle(input_file).filter(like="Pose")
    df = df.filter(regex="^((?!Coxa).)*$")
    
    x_coords = df.filter(like="_x").values
    y_coords = df.filter(like="_y").values
    z_coords = df.filter(like="_z").values
    
    data = np.hstack([x_coords, y_coords, z_coords]).astype("float32")

    np.save(output_file, data)
