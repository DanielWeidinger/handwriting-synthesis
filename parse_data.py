import os
import numpy as np
from data.extract import collect_data, get_stroke_sequence
from data import stroke_utils

print('traversing data directory...')
stroke_fnames, transcriptions, writer_ids = collect_data()


print('dumping to numpy arrays...')

x = np.zeros([len(stroke_fnames), stroke_utils.MAX_STROKE_LEN, 3],
             dtype=np.float32)

x_in = np.zeros([len(stroke_fnames), stroke_utils.MAX_STROKE_LEN, 3],
                dtype=np.float32)
x_out = np.zeros([len(stroke_fnames), stroke_utils.MAX_STROKE_LEN, 3],
                 dtype=np.float32)
x_len = np.zeros([len(stroke_fnames)], dtype=np.int16)
c = np.zeros([len(stroke_fnames), stroke_utils.MAX_CHAR_LEN], dtype=np.int8)
c_len = np.zeros([len(stroke_fnames)], dtype=np.int8)
w_id = np.zeros([len(stroke_fnames)], dtype=np.int16)
valid_mask = np.zeros([len(stroke_fnames)], dtype=np.bool_)

for i, (stroke_fname, c_i, w_id_i) in enumerate(zip(stroke_fnames, transcriptions, writer_ids)):
    if i % 200 == 0:
        print(i, '\t', '/', len(stroke_fnames))

    x_i = get_stroke_sequence(stroke_fname)
    valid_mask[i] = ~np.any(np.linalg.norm(x_i[:, :2], axis=1) > 60)

    x[i, :len(x_i), :] = x_i
    x_in[i, :len(x_i)-1, :] = x_i[:-1]
    x_out[i, :len(x_i)-1, :] = x_i[1:]
    x_len[i] = len(x_i)

    c[i, :len(c_i)] = c_i
    c_len[i] = len(c_i)

    w_id[i] = w_id_i

if not os.path.isdir('data/processed'):
    os.makedirs('data/processed')


# c_one_hot = np.zeros((*c.shape, np.uint(stroke_utils.alphabet_len)))
# idx = np.arange(c.shape[1])
# for i in range(len(c)):
#     current = c[i]
#     c_one_hot[i, idx, current] = 1
# c=c_one_hot

np.save('data/processed/x.npy', x[valid_mask])
np.save('data/processed/x_in.npy', x_in[valid_mask])
np.save('data/processed/x_out.npy', x_out[valid_mask])
np.save('data/processed/x_len.npy', x_len[valid_mask])
np.save('data/processed/c.npy', c[valid_mask])
np.save('data/processed/c_len.npy', c_len[valid_mask])
np.save('data/processed/w_id.npy', w_id[valid_mask])
