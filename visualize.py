import matplotlib.pyplot as plt
import numpy as np
import data.stroke_utils as su


x = np.load('data/processed/x.npy')
x_len = np.load('data/processed/x_len.npy')

sample_idx = 20
current_sample = su.offsets_to_coords(x[sample_idx][:x_len[sample_idx]])
# current_sample = su.offsets_to_coords(current_sample)
print(current_sample)

c = np.load('data/processed/c.npy')
c_len = np.load('data/processed/c_len.npy')
c_len[sample_idx]

# print(current_sample[:, 0])
# print(current_sample[:, 1])
stroke = []
for x, y, eos in current_sample:
    stroke.append((x, y))
    if eos == 1:
        coords = list(zip(*stroke))
        plt.plot(coords[0], coords[1], 'k')
        stroke = []

# plt.plot(current_sample[:, 0], current_sample[:, 1])
title_num = c[sample_idx][:c_len[sample_idx]]
title = "".join([chr(su.num_to_alpha[x]) for x in title_num])
plt.title(title)
plt.show()
# np.load('data/processed/w_id.npy')
