import numpy as np

data =  np.load("./data.npy")
label = np.load("./label.npy")

indices = np.arange(data.shape[0])
np.random.shuffle(indices)

shuffle_data = data[indices]
shuffle_label = label[indices]

np.save("shuffle_data",shuffle_data)
np.save("shuffle_label",shuffle_label)
