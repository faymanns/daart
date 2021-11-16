import os.path
import pickle

import numpy as np
import pandas as pd
import scipy.signal

import utils2p
import utils2p.synchronization
import utils_ballrot
import utils_video
import utils_video.generators


skip_existing = False

trial_dirs = []
with open("trials.txt", "r") as f:
    for line in f:
        if line[:1] == "#":
            continue
        trial_dirs.append(line.rstrip())

def hysteresis_filter(seq, n=5, n_false=None):
    """
    This function implements a hysteresis filter for boolean sequences.
    The state in the sequence only changes if n consecutive element are in a different state.

    Parameters
    ----------
    seq : 1D np.array of type boolean
        Sequence to be filtered.
    n : int, default=5
        Length of hysteresis memory.
    n_false : int, optional, default=None
        Length of hystresis memory applied for the false state.
        This means the state is going to change to false when it encounters
        n_false consecutive entries with value false.
        If None, the same value is used for true and false.

    Returns
    -------
    seq : 1D np.array of type boolean
        Filtered sequence.
    """
    if n_false is None:
        n_false = n
    #seq = seq.astype(np.bool)
    state = seq[0]
    start_of_state = 0
    memory = 0

    current_n = n
    if state:
        current_n = n_false
    
    for i in range(len(seq)):
        if state != seq[i]:
            memory += 1
        elif memory < current_n:
            memory = 0
            continue
        if memory == current_n:
            seq[start_of_state:i - current_n + 1] = state
            start_of_state = i - current_n + 1
            state = not state
            if state:
                current_n = n_false
            else:
                current_n = n
            memory = 0
    seq[start_of_state:] = state
    return seq

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
        output_file = f"/mnt/internal_hdd/aymanns/daart_heuristic_videos/{beh}_{descriptor}.mp4"
        grid_generator = utils_video.generators.grid(generators, allow_different_length=True)
        utils_video.make_video(output_file, grid_generator, 100, n_frames=max_length + 200)


filt = scipy.signal.butter(10, 4, btype="lowpass", fs=100, output="sos")

# the state ordering should be the same between the hand and heuristic labels
state_mapping = {
    0: 'background',
    1: 'resting',
    2: 'walking',
    3: 'head_grooming',
    4: 'foreleg_grooming',
    5: 'hind_grooming',
    }

#from matplotlib import pyplot as plt
#fig, axes = plt.subplots(2, 1, sharex=True)
for trial_dir in trial_dirs:
    #if "003" not in trial_dir:
    #    continue
    descriptor = trial_dir.replace("/mnt/data2/FA/", "").rstrip("/").replace("/", "_")
    output_file = f"labels-heuristic/{descriptor}_labels.pkl"
    
    print(output_file)

    if skip_existing and os.path.isfile(output_file):
        continue

    input_file = os.path.join(trial_dir, "behData/images/df3d/post_processed.pkl")
    df = pd.read_pickle(input_file).filter(like="Pose")

    finite_difference_coefficients = np.array([1/280, -4/105, 1/5, -4/5, 0, 4/5, -1/5, 4/105, -1/280])
    # step size (spacing between elements)
    h = 0
    kernel = np.zeros(len(finite_difference_coefficients) * (1 + h) - h)
    kernel[::(h+1)] = finite_difference_coefficients

    motion_energy_df = pd.DataFrame()
    direction_df = pd.DataFrame()
    front_height_df = pd.DataFrame()
    for pair in ("F", "M", "H"):
        for side in ("R", "L"):
            for joint in ("Femur", "Tibia", "Tarsus", "Claw"):
                derivatives = []
                for axis in ("x", "y", "z"):
                    col_name = f"Pose__{side}{pair}_leg_{joint}_{axis}"
                    deriv = np.convolve(df[col_name], kernel, mode="same")
                    #deriv = scipy.signal.sosfiltfilt(filt, deriv)
                    derivatives.append(deriv) 

                    if pair == "F" and axis == "z" and joint == "Tarsus":
                        front_height_df[f"{side}"] = df[col_name]

                derivatives = np.array(derivatives)
                #motion_energy_df[f"{side}{pair}_{joint}"] = scipy.signal.sosfiltfilt(filt, np.sum(derivatives ** 2, axis=0))
                motion_energy_df[f"{side}{pair}_{joint}"] = scipy.signal.sosfiltfilt(filt, np.sum(np.abs(derivatives), axis=0))
                #motion_energy_df[f"{side}{pair}_{joint}"] = np.sum(scipy.signal.sosfiltfilt(filt, derivatives) ** 2, axis=0)
                direction_vectors = np.abs(derivatives) / np.linalg.norm(derivatives, axis=0)
                alpha = np.arctan(direction_vectors[0, :] / direction_vectors[2, :])
                beta = np.arctan(direction_vectors[0, :] / direction_vectors[1, :])
                direction_df[f"{side}{pair}_{joint}_alpha"] = scipy.signal.sosfiltfilt(filt, alpha)
                direction_df[f"{side}{pair}_{joint}_beta"] = scipy.signal.sosfiltfilt(filt, beta)
                direction_df[f"{side}{pair}_{joint}_x"] = derivatives[0, :]
                direction_df[f"{side}{pair}_{joint}_y"] = derivatives[1, :]
                direction_df[f"{side}{pair}_{joint}_z"] = derivatives[2, :]

    total_motion_energy = np.sum(motion_energy_df.values, axis=1)
    front_motion_energy = np.sum(motion_energy_df.filter(like="F_"), axis=1)
    hind_motion_energy = np.sum(motion_energy_df.filter(like="H_"), axis=1)

    front_height_df["height"] = scipy.signal.sosfiltfilt(filt, (front_height_df["R"] + front_height_df["L"]) / 2)

    fictrac_file = utils2p.find_fictrac_file(trial_dir, most_recent=True)
    fictrac_data = utils_ballrot.load_fictrac(fictrac_file)

    vel = fictrac_data["delta_rot_forward"]
    filtered_vel = scipy.signal.sosfiltfilt(filt, vel)
    
    binary_seq_walking = (filtered_vel > 0.5)
    binary_seq_walking = hysteresis_filter(binary_seq_walking, n=100, n_false=50)
    
    #binary_seq_backward_walking = (filtered_vel < 0)
    #binary_seq_backward_walking = hysteresis_filter(binary_seq_backward_walking, n=100, n_false=50)

    #binary_seq_remaining = np.logical_and(~binary_seq_walking, ~binary_seq_backward_walking)
    binary_seq_remaining = ~binary_seq_walking
    binary_seq_resting = np.logical_and(binary_seq_remaining, total_motion_energy < 0.3)
    binary_seq_resting = hysteresis_filter(binary_seq_resting, n=50)
    #make_video_based_on_binary_seq(binary_seq_resting, trial_dir, "resting", descriptor)
    #continue
    
    #hand_labels_file = f"labels-hand/{descriptor}_labels.csv"
    #hand_labels = np.genfromtxt(hand_labels_file, delimiter=",", dtype=int, encoding=None)
    #hand_labels = hand_labels[1:, 1:]
    #grooming_labels = np.logical_or(hand_labels[:, 3], hand_labels[:, 4])
    #grooming_labels = np.logical_or(grooming_labels, hand_labels[:, 5])
    #resting_labels = hand_labels[:, 1]
    #hindgrooming_labels = np.logical_or(hand_labels[:, 5], hand_labels[:, 6])
    #print(sum(resting_labels))
    #axes[0].hist(front_motion_energy[grooming_labels], density=True, alpha=0.1)
    #axes[1].hist(hind_motion_energy[grooming_labels], density=True, alpha=0.1)
    #axes[0].hist(total_motion_energy[resting_labels], density=True, alpha=0.1)
    #axes[1].hist(total_motion_energy[grooming_labels], density=True, alpha=0.1)
    #axes[0].hist(front_motion_energy[hindgrooming_labels], density=True, alpha=0.1)
    #axes[1].hist(hind_motion_energy[hindgrooming_labels], density=True, alpha=0.1)
    #hand_labels = pd.read_csv(hand_labels_file, index_col=0)
    #foreleg_grooming_labels = hand_labels["foreleg_grooming"].values.astype(bool)
    #eye_grooming_labels = hand_labels["eye_grooming"].values.astype(bool)
    #axes[0].hist(front_height_df["height"][foreleg_grooming_labels], alpha=0.1)
    #axes[1].hist(front_height_df["height"][eye_grooming_labels], alpha=0.1)
    #continue

    binary_seq_remaining = np.logical_and(binary_seq_remaining, ~binary_seq_resting)
    binary_seq_grooming = np.logical_and(front_motion_energy > 0.2, hind_motion_energy < 0.2)
    binary_seq_grooming = np.logical_and(binary_seq_remaining, binary_seq_grooming)
    binary_seq_grooming = hysteresis_filter(binary_seq_grooming, n=50)
    #make_video_based_on_binary_seq(binary_seq_grooming, trial_dir, "grooming", descriptor)
    #continue

    binary_seq_eye_grooming = np.logical_and(binary_seq_grooming, front_height_df["height"].values > 0.05)
    binary_seq_eye_grooming = hysteresis_filter(binary_seq_eye_grooming, n=50)
    #make_video_based_on_binary_seq(binary_seq_eye_grooming, trial_dir, "eye_grooming", descriptor)
    #continue
    binary_seq_foreleg_grooming = np.logical_and(binary_seq_grooming, front_height_df["height"].values < 0.05)
    binary_seq_foreleg_grooming = hysteresis_filter(binary_seq_foreleg_grooming, n=50)
    #make_video_based_on_binary_seq(binary_seq_foreleg_grooming, trial_dir, "foreleg_grooming", descriptor)
    #continue

    binary_seq_remaining = np.logical_and(binary_seq_remaining, ~binary_seq_grooming)
    binary_seq_hindgrooming = np.logical_and(front_motion_energy < 0.2, hind_motion_energy > 0.2)
    binary_seq_hindgrooming = np.logical_and(binary_seq_remaining, binary_seq_hindgrooming)
    binary_seq_hindgrooming = hysteresis_filter(binary_seq_hindgrooming, n=50)
    #make_video_based_on_binary_seq(binary_seq_hindgrooming, trial_dir, "hindgrooming", descriptor)
    #continue
    
    states = np.zeros(df.shape[0], dtype=int)

    states[binary_seq_resting] = 1
    states[binary_seq_walking] = 2
    states[binary_seq_eye_grooming] = 3
    states[binary_seq_foreleg_grooming] = 4
    states[binary_seq_hindgrooming] = 5
    #states[binary_seq_abdominal_grooming] = 5
    #states[binary_seq_hindleg_grooming] = 6
    #states[binary_seq_antennal_grooming] = 
    #states[binary_seq_backward_walking] = 

    print(states.shape)
    continue
    data = {'states': states, 'state_labels': state_mapping}
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)
    
    
    
    
    
    #direction_df = direction_df.loc[binary_seq_grooming, :]
    #binary_seq_grooming = np.zeros(direction_df.shape[0], dtype=bool)
    #date = utils.get_date(trial_dir)
    #genotype = utils.get_genotype(trial_dir)
    #fly = utils.get_fly_number(trial_dir)
    #trial = utils.get_trial_number(trial_dir)
    #trial_annotations = annotation_df.loc[(date, genotype, fly, trial, slice(None)), :]
    #frames = trial_annotations.index.to_frame(name=["Date", "Genotype", "Fly", "Trial", "Frame"])["Frame"].values
    #trial_annotations = trial_annotations.reset_index()
    #import utils_video
    #import utils_video.generators
    #video_file = os.path.join(trial_dir, "behData/images/camera_1.mp4")
    #for beh in ("antennal_grooming", "eye_grooming", "foreleg_grooming"):
    #    #beh_frames = sorted(frames[trial_annotations["Behaviour"] == beh])
    #    beh_frames = sorted(trial_annotations.loc[trial_annotations["Behaviour"] == beh, "Frame"])
    #    if len(beh_frames) == 0:
    #        continue
    #    locs = np.concatenate(([0,], np.where(np.diff(beh_frames) > 1)[0] + 1, [len(beh_frames) - 1,]))
    #    for i, (start, stop) in enumerate(zip(locs[:-1], locs[1:])):
    #        print(start, stop, beh_frames[start], beh_frames[stop - 1])
    #        gen = utils_video.generators.video(video_file, start=beh_frames[start])
    #        utils_video.make_video(f"{beh}_{i}_{descriptor}.mp4", gen, 100, n_frames=beh_frames[stop - 1] - beh_frames[start])

    #continue
    #from matplotlib import pyplot as plt
    #pair = "F"
    #for side in ("R", "L"):
    #    for joint in ("Femur", "Tibia", "Tarsus", "Claw"):
    #        for beh in ("antennal_grooming", "eye_grooming", "foreleg_grooming"):
    #            bool_index = trial_annotations["Behaviour"] == beh
    #            plt.scatter(direction_df.loc[frames[bool_index], f"{side}{pair}_{joint}_alpha"].values,
    #                        direction_df.loc[frames[bool_index], f"{side}{pair}_{joint}_beta"].values,
    #                        label=beh
    #                       )
    #        plt.legend()
    #        plt.savefig(f"grooming_dir_{side}{pair}_{joint}_{descriptor}.pdf")
    #        plt.close()
    #continue
    #colors = ["blue", "orange", "green"]
    #fig, axes = plt.subplots(4, 2)
    #for joint_idx, joint in enumerate(("Femur", "Tibia", "Tarsus", "Claw")):
    #    for beh_idx, beh in enumerate(("antennal_grooming", "eye_grooming", "foreleg_grooming")):
    #        for side in ("R", "L"):
    #            bool_index = trial_annotations["Behaviour"] == beh
    #            axes[joint_idx, 0].scatter(direction_df.loc[frames[bool_index], f"{side}{pair}_{joint}_x"].values,
    #                        direction_df.loc[frames[bool_index], f"{side}{pair}_{joint}_y"].values,
    #                        label=beh,
    #                        color=colors[beh_idx]
    #                       )
    #            axes[joint_idx, 1].scatter(direction_df.loc[frames[bool_index], f"{side}{pair}_{joint}_x"].values,
    #                        direction_df.loc[frames[bool_index], f"{side}{pair}_{joint}_z"].values,
    #                        label=beh,
    #                        color=colors[beh_idx]
    #                       )
    #plt.legend()
    #plt.savefig(f"grooming_dir_{descriptor}.pdf")
    #plt.close()
    #continue
#axes[0].set_xlim((-0.1, 0.2))
#plt.savefig("hist.pdf")
