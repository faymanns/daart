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
with open(
    "/mnt/internal_hdd/aymanns/ABO_data_processing/behaviour_classifier.pkl",
    "rb",
) as f:
    clf = pickle.load(f)
with open(
    "/mnt/internal_hdd/aymanns/ABO_data_processing/label_encoder.pkl", "rb"
) as f:
    le = pickle.load(f)

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
    #pose_df = pose_df.filter(like="Pose")
    #pose_df = pose_df.filter(regex="^((?!Coxa).)*$")
    #
    #x_coords = pose_df.filter(like="_x").values
    #y_coords = pose_df.filter(like="_y").values
    #z_coords = pose_df.filter(like="_z").values
    #
    #data = np.hstack([x_coords, y_coords, z_coords]).astype("float32")
    angle_df = pose_df.filter(like="Angle")
    data = angle_df.values.astype("float32")

    np.save(f"markers/{descriptor}_labeled.npy", data)

    trial_annotations = annotation_df.loc[(row["Date"], row["Genotype"], row["Fly"], row["Trial"], slice(None)), :]

    hand_labels_df = pd.DataFrame()
    #for beh in ["background", "resting", "walking", "grooming", "hindgrooming"]:
    #for beh in ["background", "resting", "walking", "eye_grooming", "antennal_grooming", "foreleg_grooming", "abdominal_grooming", "hindleg_grooming", "backward_walking"]:
    for beh in ["background", "resting", "walking", "eye_grooming", "foreleg_grooming", "abdominal_grooming", "hindleg_grooming",]:
        hand_labels_df[beh] = np.zeros(pose_df.shape[0], dtype=int)

    #print(trial_annotations["Behaviour"].unique())

    frames = trial_annotations.index.to_frame(name=["Date", "Genotype", "Fly", "Trial", "Frame"])["Frame"]

    bool_index = trial_annotations["Behaviour"] == "resting"
    hand_labels_df.loc[frames[bool_index], "resting"] = 1

    bool_index = trial_annotations["Behaviour"] == "walking"
    hand_labels_df.loc[frames[bool_index], "walking"] = 1

    bool_index = trial_annotations["Behaviour"].isin(("eye_grooming", "antennal_grooming"))#, "foreleg_grooming"))
    hand_labels_df.loc[frames[bool_index], "eye_grooming"] = 1
    
    #bool_index = trial_annotations["Behaviour"] == "eye_grooming"
    #hand_labels_df.loc[frames[bool_index], "grooming"] = 1

    #bool_index = trial_annotations["Behaviour"] == "antennal_grooming"
    #hand_labels_df.loc[frames[bool_index], "grooming"] = 1

    bool_index = trial_annotations["Behaviour"] == "foreleg_grooming"
    hand_labels_df.loc[frames[bool_index], "foreleg_grooming"] = 1

    #bool_index = trial_annotations["Behaviour"].isin(("abdominal_grooming", "hindleg_grooming"))
    #hand_labels_df.loc[frames[bool_index], "hindgrooming"] = 1

    bool_index = trial_annotations["Behaviour"] == "abdominal_grooming"
    hand_labels_df.loc[frames[bool_index], "abdominal_grooming"] = 1

    bool_index = trial_annotations["Behaviour"] == "hindleg_grooming"
    hand_labels_df.loc[frames[bool_index], "hindleg_grooming"] = 1
    #
    #bool_index = trial_annotations["Behaviour"] == "backward_walking"
    #hand_labels_df.loc[frames[bool_index], "backward_walking"] = 1
    
    # the state ordering should be the same between the hand and heuristic labels
    #state_mapping = {
    #    0: 'background',
    #    1: 'resting',
    #    2: 'walking',
    #    3: 'grooming',
    #    4: 'hindgrooming',
    #    }
    state_mapping = {
            0: 'background',
            1: 'resting',
            2: 'walking',
            3: 'eye_grooming',
            4: 'foreleg_grooming',
            5: 'abdominal_grooming',
            6: 'hindleg_grooming',
        }
    #state_mapping = {
    #    0: 'background',
    #    1: 'resting',
    #    2: 'walking',
    #    3: 'eye_grooming',
    #    4: 'antennal_grooming',
    #    5: 'foreleg_grooming',
    #    6: 'abdominal_grooming',
    #    7: 'hindleg_grooming',
    #    8: 'backward_walking',
    #    }

    #wavelet_df = pd.read_pickle(os.path.join(trial_dir, "behData/images/df3d/angle_wavelets.pkl"))
    #feature_columns = [col for col in wavelet_df.columns if "Coeff" in col]
    #X = wavelet_df[feature_columns].values
    #y_pred = clf.predict(X)
    #prediction = le.inverse_transform(y_pred)
    #mask = states == 0
    #for num, beh in state_mapping.items():
    #    bool_index = np.logical_and(mask, prediction == beh)
    #    states[bool_index] = num

    background = hand_labels_df.sum(axis=1) == 0
    hand_labels_df.loc[background, "background"] = 1
    #hand_labels_df.loc[background, :] = -1
    hand_labels_df.to_csv(f"labels-hand/{descriptor}_labels.csv")
    
    print(hand_labels_df.values.shape)
    print(hand_labels_df.columns)
    states = np.argmax(hand_labels_df.values, axis=1)
    data = {'states': states, 'state_labels': state_mapping}
    with open(f"labels-heuristic/{descriptor}_labels.pkl", 'wb') as f:
        pickle.dump(data, f)

    #print(sum(np.logical_xor(states == 0, hand_labels_df["background"])))

#with open("descriptors.txt", "w") as f:
#    for d in descriptors:
#        f.write(d)
#        f.write("\n")
