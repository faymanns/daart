import numpy as np


data = np.load("markers/201016_G23xU1_Fly1_020_coronal_labeled.npy")
print(np.where(np.isnan(data)))
