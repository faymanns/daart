import os.path
import itertools

import pandas as pd
import numpy as np
import utils_video
import utils_video.generators
import utils2p.synchronization


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

def make_video_based_on_binary_seq(binary_seq, trial_dir, beh, descriptor):
    if "ref" in trial_dir:
        video_file = os.path.join(trial_dir[:-9], "behData/images/camera_1.mp4")
    else:
        video_file = os.path.join(trial_dir, "behData/images/camera_1.mp4")
    event_based_indices, event_numbers = utils2p.synchronization.event_based_frame_indices(binary_seq) 
    generators = []
    max_length = 0
    for event_number in np.unique(event_numbers):
        if event_number == -1:
            continue
        event_mask = (event_numbers == event_number)
        event_length = np.max(event_based_indices[event_mask]) + 1 
        if event_length > max_length:
            max_length = event_length
        start = np.where(np.logical_and(event_mask, event_based_indices == 0))[0][0] - 200
        generator = utils_video.generators.video(video_file, start=start)
        dot_mask = event_mask[start:]
        dot_mask[:200] = 0
        generator = utils_video.generators.add_stimulus_dot(generator, dot_mask)
        generators.append(generator)

    print("Number of generators:", len(generators))
    print("Max event length:", max_length)
    if len(generators) > 0:
        output_file = f"/mnt/internal_hdd/aymanns/daart_inspect_old_annotations_videos/{beh}_{descriptor}.mp4"
        grid_generator = utils_video.generators.grid(generators, allow_different_length=True)
        utils_video.make_video(output_file, grid_generator, 100, n_frames=max_length + 200)


annotation_df = pd.read_pickle("/mnt/internal_hdd/aymanns/ABO_data_processing/annotations_df.pkl")
index_df = annotation_df.index.to_frame(name=["Date", "Genotype", "Fly", "Trial", "Frame"])
annotated_trials = index_df.drop_duplicates(subset=["Date", "Genotype", "Fly", "Trial"])

for index, row in annotated_trials.iterrows():

    trial_dir = get_trial_dir(row["Date"], row["Genotype"], row["Fly"], row["Trial"])
    print(trial_dir)
    
    descriptor = trial_dir.replace("/mnt/data2/FA/", "").replace("/mnt/data/FA/", "").replace("/mnt/lab_server/AYMANNS_Florian/Experimental_data/", "").rstrip("/").replace("/", "_")
    
    hand_labels_df = pd.read_csv(f"labels-hand/{descriptor}_labels.csv", index_col=0)
    video_file = os.path.join(trial_dir, "behData/images/camera_1.mp4")

    for beh in hand_labels_df.columns:
        print(beh)
        if beh == "background":
            continue
        binary_seq = hand_labels_df[beh].values
        make_video_based_on_binary_seq(binary_seq, trial_dir, beh, descriptor)
