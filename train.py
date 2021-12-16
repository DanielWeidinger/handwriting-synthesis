import tensorflow as tf
import numpy as np
import time
from tqdm import tqdm

from model.transformer import Transformer

NUM_EPOCHS = 1
model = Transformer()

strokes_in = np.load('data/processed/x_in.npy')
strokes_out = np.load('data/processed/x_out.npy')
chars = np.load('data/processed/c.npy')

BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 100
TRAIN_SPLIT = 0.7

train_len = int(len(strokes_in)*TRAIN_SPLIT)
# TODO: wrong x and y naming, coz that are not the labels
x_train = (strokes_in[train_len:], strokes_out[train_len:])
y_train = chars[train_len:]
x_val = (strokes_in[:train_len], strokes_out[:train_len])
y_val = chars[:train_len]

train_dataset = tf.data.Dataset.from_tensor_slices((*x_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((*x_val, y_val))

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

start_time = time.time()
loss, loss_coords, loss_eos = tf.constant(0)
for e in range(NUM_EPOCHS):
    for step, (strokes_in, strokes_out, chars) in tqdm(enumerate(train_dataset), total=len(train_dataset)):
        # loss, loss_coords, loss_eos =
        loss, loss_coords, loss_eos = model.train_step(
            chars, strokes_in, strokes_out)

    print('Epoch {} Loss {:.4f} Coords: {:.4f} EoS: {:.4f}'.format(
        e + 1, loss.numpy(), loss_coords.numpy(), loss_eos.numpy()))

    end_time = time.time()
    print('Average elapsed time: {:.2f}s'.format(
        (end_time - start_time) / (e + 1)))
