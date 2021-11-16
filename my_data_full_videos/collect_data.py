import os.path
import pickle

import pandas as pd
import numpy as np


n_frames = 540001
#trial_dir = "/mnt/data2/FA/211029_Ci1xG23/Fly1/003_coronal"
#hand_labels = pd.read_csv("/mnt/internal_hdd/aymanns/deepethogram/DN_paper_deepethogram/DATA/211029_Fly1_003/211029_Fly1_003_labels.csv", index_col=0)
#n_frames = 7520
#trial_dir = "/mnt/data2/FA/211027_Ci1xG23/Fly2/003_coronal"
#hand_labels = pd.read_csv("/mnt/internal_hdd/aymanns/deepethogram/DN_paper_deepethogram/DATA/211027_Fly2_003/211027_Fly2_003_labels.csv", index_col=0)
#n_frames = 11125
#trial_dir = "/mnt/data2/FA/211026_Ci1xG23/Fly3/003_coronal"
#hand_labels = pd.read_csv("/mnt/internal_hdd/aymanns/deepethogram/DN_paper_deepethogram/DATA/211026_Fly3_003/211026_Fly3_003_labels.csv", index_col=0)
#n_frames = 6025
#trial_dir = "/mnt/data2/FA/210910_Ci1xG23/Fly2/003_coronal"
#hand_labels = pd.read_csv("/mnt/internal_hdd/aymanns/deepethogram/DN_paper_deepethogram/DATA/210910_Fly2_003/210910_Fly2_003_labels.csv", index_col=0)
#n_frames = 6200
trial_dir = "/mnt/data2/FA/210830_Ci1xG23/Fly1/003_coronal"
hand_labels = pd.read_csv("/mnt/internal_hdd/aymanns/deepethogram/DN_paper_deepethogram/DATA/210830_Fly1_003/210830_Fly1_003_labels.csv", index_col=0)
#n_frames = 6036

descriptor = trial_dir.replace("/mnt/data2/FA/", "").rstrip("/").replace("/", "_")

input_file = os.path.join(trial_dir, "behData/images/df3d/post_processed.pkl")
df = pd.read_pickle(input_file)

data = df.filter(like="Angle").values.astype("float32")
data = data[:n_frames + 1]

output_file = f"markers/{descriptor}_labeled.npy"
print(data.shape)
np.save(output_file, data)


hand_labels = hand_labels.loc[:n_frames, :]
hand_labels["head_grooming"] = (hand_labels["eye_grooming"] + hand_labels["antennal_grooming"]).clip(upper=1)
hand_labels["hind_grooming"] = (hand_labels["abdominal_grooming"] + hand_labels["hindleg_grooming"]).clip(upper=1)
hand_labels = hand_labels.drop(labels=["eye_grooming", "antennal_grooming", "backward_walking", "abdominal_grooming", "hindleg_grooming"], axis=1)
background = hand_labels.sum(axis=1) < 0
hand_labels.loc[background, :] = 0
hand_labels.loc[background, "background"] = 1
print(hand_labels.shape)
hand_labels = hand_labels[["background", "resting", "walking", "head_grooming", "foreleg_grooming", "hind_grooming"]]
hand_labels.to_csv(f"labels-hand/{descriptor}_labels.csv")

#states = np.argmax(hand_labels.values, axis=1)
#state_mapping = {
#        0: 'background',
#        1: 'resting',
#        2: 'walking',
#        3: 'eye_grooming',
#        4: 'foreleg_grooming',
#        5: 'abdominal_grooming',
#        6: 'hindleg_grooming',
#    }
#heuristic_labels = {'states': states, 'state_labels': state_mapping}
#print(states.shape)
#print(np.min(states), np.max(states))
#with open(f"labels-heuristic/{descriptor}_labels.pkl", 'wb') as f:
#    pickle.dump(heuristic_labels, f)
