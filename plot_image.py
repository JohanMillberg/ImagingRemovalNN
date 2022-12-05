import numpy as np
import matplotlib.pyplot as plt
import sys, getopt

filepath_label = f"images/labels/{sys.argv[1]}"
filepath_data = f"images/data/{sys.argv[1]}"

label = np.load(filepath_label)
data = np.load(filepath_data)

y_range = range(81, 81 + 350)
x_range = range(25, 25 + 175)
indices = [y*512 + x for y in y_range for x in x_range]
label = label[indices].reshape((350, 175))
    
data = data.reshape((350, 175))

data = (data - np.min(data)) / np.ptp(data)
label = (label - np.min(label)) / np.ptp(label)

fig, ax = plt.subplots(1, 2)
ax[0].imshow(label.squeeze(), cmap="gray")
ax[0].set_title("Generated fracture image")
ax[1].imshow(data.squeeze(), cmap="gray")
ax[1].set_title("Result of the imaging algorithm")
plt.show()
