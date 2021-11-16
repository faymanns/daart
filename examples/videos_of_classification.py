import os

import pandas as pd
import numpy as np

import utils2p.synchronization
import utils_video
import utils_video.generators


trial_dirs = []
with open("../my_data/trials.txt", "r") as f:
    for line in f:
        if line[:1] == "#":
            continue
        trial_dirs.append(line.rstrip())


class_names = np.array(['background', 'resting', 'walking', 'eye_grooming', 'antennal_grooming', 'foreleg_grooming', 'abdominal_grooming', 'hindleg_grooming', "backward_walking"])
#class_names = np.array(['background', 'resting', 'walking', 'grooming', 'hindgrooming'])
#class_names = np.array(['resting', 'walking', 'grooming', 'hindgrooming'])

for trial_dir in trial_dirs:
    descriptor = trial_dir.replace("/mnt/data2/FA/", "").rstrip("/").replace("/", "_")

    for beh in class_names:
        print(beh)

        #output_file = f"/mnt/internal_hdd/aymanns/daart_behaviour_videos_8classes_5000/{beh}_{descriptor}.mp4"
        #output_file = f"/mnt/internal_hdd/aymanns/daart_behaviour_videos_with_heuristic_labels/{beh}_{descriptor}.mp4"
        output_file = f"/mnt/internal_hdd/aymanns/daart_behaviour_videos_full_videos/{beh}_{descriptor}.mp4"
        generators = []
        max_length = 0

        print(trial_dir)
        video_file = os.path.join(trial_dir, "behData/images/camera_1.mp4")

        predictions = pd.read_pickle(os.path.join(trial_dir, "behData/images/df3d/behaviour_predictions_daart.pkl"))

        binary_seq = predictions["Prediction"].values == beh
        if sum(binary_seq) == 0:
            continue
        probabilities = list(map(str, predictions[f"Probability {beh}"].values))

        event_based_indices, event_numbers = utils2p.synchronization.event_based_frame_indices(binary_seq) 

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
            generator = utils_video.generators.add_text_PIL(generator, probabilities[start:])
            generators.append(generator)

        print("Number of generators:", len(generators))
        print("Max event length:", max_length)
        if len(generators) > 0:
            grid_generator = utils_video.generators.grid(generators, allow_different_length=True)
            utils_video.make_video(output_file, grid_generator, 100, n_frames=max_length + 200)
