import numpy as np
import data.stroke_utils as su

c = np.load('data/processed/c.npy')
c_len = np.load('data/processed/c_len.npy')

c_one_hot = np.zeros((*c.shape, np.uint(su.alphabet_len)))
idx = np.arange(c.shape[1])
for i in range(len(c)):
    current = c[i]
    c_one_hot[i, idx, current] = 1
