import os.path

import numpy as np
import pandas as pd


trial_dirs = []
with open("trials.txt", "r") as f:
    for line in f:
        if line[:1] == "#":
            continue
        trial_dirs.append(line.rstrip())

skip_existing = False

for trial_dir in trial_dirs:

    descriptor = trial_dir.replace("/mnt/data2/FA/", "").rstrip("/").replace("/", "_")
    print(descriptor)
    output_file = f"labels-hand/{descriptor}_labels.csv"

    if skip_existing and os.path.isfile(output_file):
        continue

    input_file = os.path.join(trial_dir, "behData/images/df3d/post_processed.pkl")
    df = pd.read_pickle(input_file).filter(like="Pose")
    df = df.filter(regex="^((?!Coxa).)*$")

    hand_labels_df = pd.DataFrame()
    #for beh in ["background", "resting", "walking", "grooming", "hindgrooming"]:
    for beh in ["background", "resting", "walking", "eye_grooming", "antennal_grooming", "foreleg_grooming", "abdominal_grooming", "hindleg_grooming", "backward_walking"]:
        hand_labels_df[beh] = np.zeros(df.shape[0], dtype=int)

    hand_labels_df["background"] = 1
    hand_labels_df.to_csv(output_file)
