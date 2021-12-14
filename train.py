import tensorflow as tf
import numpy as np
import time

from model.transformer import Transformer

NUM_EPOCHS = 100
model = Transformer()

strokes = np.load('data/processed/x.npy')
chars = np.load('data/processed/c.npy')

BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 100
TRAIN_SPLIT = 0.7

train_len = len(strokes)*TRAIN_SPLIT
x_train = strokes[train_len:]
y_train = chars[train_len:]
x_val = strokes[:train_len]
y_val = chars[:train_len]

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

start_time = time.time()
for e in range(NUM_EPOCHS):
    for step, (strokes, chars) in enumerate(train_dataset):
        loss = model.train_step(source_seq, target_seq_in,
                                target_seq_out)

    print('Epoch {} Loss {:.4f}'.format(
          e + 1, loss.numpy()))

    if (e + 1) % 10 == 0:
        end_time = time.time()
        print('Average elapsed time: {:.2f}s'.format(
            (end_time - start_time) / (e + 1)))
        # try:
        #     predict()
        # except Exception as e:
        #     print(e)
        #     continue
