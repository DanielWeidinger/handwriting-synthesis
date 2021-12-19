import tensorflow as tf
import numpy as np
import time
from tqdm import tqdm

from model.training import avg_error_distance, eos_accuracy
from model.transformer import Transformer

EPOCHS = 10
model = Transformer()

strokes_in = np.load('data/processed/x_in.npy')
strokes_out = np.load('data/processed/x_out.npy')
# TODO: remove if unnecessary
strokes_len = np.load('data/processed/x_len.npy')
chars = np.load('data/processed/c.npy')

BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 100
TRAIN_SPLIT = 0.8

tar_mask = np.zeros(strokes_in.shape[:2])
for i, l in enumerate(strokes_len):
    tar_mask[i, :l] = 1

train_len = int(len(strokes_in)*TRAIN_SPLIT)
x_train = (chars[train_len:], strokes_in[train_len:])
y_train = strokes_out[train_len:]
mask_train = tar_mask[train_len:]

x_val = (chars[:train_len], strokes_in[:train_len])
y_val = strokes_out[:train_len]
mask_val = tar_mask[:train_len]

train_dataset = tf.data.Dataset.from_tensor_slices(
    (*x_train, y_train, mask_train))
test_dataset = tf.data.Dataset.from_tensor_slices((*x_val, y_val, mask_val))

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)  # TODO: use!


checkpoint_path = "./checkpoints/train"
ckpt = tf.train.Checkpoint(transformer=model,
                           optimizer=model.optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')


# metrics
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_eos_accuracy = tf.keras.metrics.Mean(name='train_accuracy')
train_avg_error_distance = tf.keras.metrics.Mean(name='train_accuracy')

for epoch in range(EPOCHS):
    start = time.time()

    train_loss.reset_states()
    train_eos_accuracy.reset_states()
    train_avg_error_distance.reset_states()

    # inp -> portuguese, tar -> english
    for batch, (inp, tar_inp, tar_out, tar_mask) in tqdm(enumerate(train_dataset), total=len(train_dataset)):
        (coords, eos), loss = model.train_step(inp, tar_inp, tar_out, tar_mask)
        predictions = tf.concat([coords, eos], -1)
        train_loss(loss)
        train_eos_accuracy(eos_accuracy(tar_out, predictions))
        train_avg_error_distance(avg_error_distance(tar_out, predictions))

        if batch % 50 == 0:
            print(
                f'Epoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f} EoS_Accuracy {train_eos_accuracy.result():.4f} Avg Coords Distance: {train_avg_error_distance.result():.4f}')

    if (epoch + 1) % 5 == 0:
        ckpt_save_path = ckpt_manager.save()
        print(f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}')

    print(
        f'Epoch {epoch + 1} Loss {train_loss.result():.4f} EoS_Accuracy {train_eos_accuracy.result():.4f} Avg Coords Distance: {train_avg_error_distance.result():.4f}')

    print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')
