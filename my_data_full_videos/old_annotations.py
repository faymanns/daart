import os.path
import itertools
import pickle

import pandas as pd
import numpy as np


def get_trial_dir(date, genotype, fly, trial):
    date = int(date)
    fly = int(fly)
    trial = int(trial)
    locations = ["/mnt/lab_server/AYMANNS_Florian/Experimental_data/", "/mnt/data/FA/", "/mnt/data2/FA/"]
    imaging_types = ["coronal", "beh"]
    trial_dir = f"{date}_{genotype}/Fly{fly}/{trial:03d}_"
    possible_trial_dirs = [os.path.join(location, trial_dir + imaging_type) for location in locations for imaging_type in imaging_types]
    existance = [os.path.isdir(d) for d in possible_trial_dirs]
    if sum(existance) > 1:
        raise ValueError(f"Trial dirs for {trial_dir} exist on multiple modalities.")
    elif not any(existance):
        raise ValueError(f"Trial dir {trial_dir} does not exist.")
    else:
        return list(itertools.compress(possible_trial_dirs, existance))[0]


annotation_df = pd.read_pickle("/mnt/internal_hdd/aymanns/ABO_data_processing/annotations_df.pkl")

index_df = annotation_df.index.to_frame(name=["Date", "Genotype", "Fly", "Trial", "Frame"])
annotated_trials = index_df.drop_duplicates(subset=["Date", "Genotype", "Fly", "Trial"])

descriptors = []

for index, row in annotated_trials.iterrows():

    trial_dir = get_trial_dir(row["Date"], row["Genotype"], row["Fly"], row["Trial"])
    print(trial_dir)
    
    descriptor = trial_dir.replace("/mnt/data2/FA/", "").replace("/mnt/data/FA/", "").replace("/mnt/lab_server/AYMANNS_Florian/Experimental_data/", "").rstrip("/").replace("/", "_")
    descriptors.append(descriptor)

    input_file = os.path.join(trial_dir, "behData/images/df3d/post_processed.pkl")
    pose_df = pd.read_pickle(input_file)
    angle_df = pose_df.filter(like="Angle")
    data = angle_df.values.astype("float32")

    np.save(f"markers/{descriptor}_labeled.npy", data)

    trial_annotations = annotation_df.loc[(row["Date"], row["Genotype"], row["Fly"], row["Trial"], slice(None)), :]

    hand_labels_df = pd.DataFrame()
    #for beh in ["background", "resting", "walking", "eye_grooming", "foreleg_grooming", "abdominal_grooming", "hindleg_grooming",]:
    for beh in ["background", "resting", "walking", "head_grooming", "foreleg_grooming",  "hind_grooming",]:
        hand_labels_df[beh] = np.zeros(pose_df.shape[0], dtype=int)

    frames = trial_annotations.index.to_frame(name=["Date", "Genotype", "Fly", "Trial", "Frame"])["Frame"]

    bool_index = trial_annotations["Behaviour"] == "resting"
    hand_labels_df.loc[frames[bool_index], "resting"] = 1

    bool_index = trial_annotations["Behaviour"] == "walking"
    hand_labels_df.loc[frames[bool_index], "walking"] = 1

    bool_index = trial_annotations["Behaviour"].isin(("eye_grooming", "antennal_grooming"))
    hand_labels_df.loc[frames[bool_index], "head_grooming"] = 1
    
    bool_index = trial_annotations["Behaviour"] == "foreleg_grooming"
    hand_labels_df.loc[frames[bool_index], "foreleg_grooming"] = 1

    bool_index = trial_annotations["Behaviour"].isin(("abdominal_grooming", "hindleg_grooming"))
    hand_labels_df.loc[frames[bool_index], "hind_grooming"] = 1

    state_mapping = {
            0: 'background',
            1: 'resting',
            2: 'walking',
            3: 'head_grooming',
            4: 'foreleg_grooming',
            5: 'hind_grooming',
        }

    background = hand_labels_df.sum(axis=1) == 0
    hand_labels_df.loc[background, "background"] = 1
    hand_labels_df.to_csv(f"labels-hand/{descriptor}_labels.csv")
    
    states = np.argmax(hand_labels_df.values, axis=1)
    data = {'states': states, 'state_labels': state_mapping}
    with open(f"labels-heuristic/{descriptor}_labels.pkl", 'wb') as f:
        pickle.dump(data, f)
