import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
from data.stroke_utils import MAX_CHAR_LEN, MAX_STROKE_LEN, encode_ascii, offsets_to_coords
from model.transformer import Transformer, sample_MDN

model = Transformer()


def sample(inputs, seed, model):
    predictions, _ = model(inputs, training=False)  # [:, -1, :]
    # sample for MDN index from mixture weights
    return sample_MDN(predictions, seed)


checkpoint_path = "./checkpoints/train"
ckpt = tf.train.Checkpoint(transformer=model,
                           optimizer=model.optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
    print('Latest checkpoint restored!!')

text = "Dei mama"
enc_input = [encode_ascii(text)]
enc_input = tf.keras.preprocessing.sequence.pad_sequences(
    enc_input, maxlen=MAX_CHAR_LEN, padding='post')

# np.zeros((1, MAX_STROKE_LEN, 3), dtype=np.float32)
output = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
output = output.write(0, [0, 0, 1])

divisor = 10
for i in tqdm(tf.range(1, MAX_STROKE_LEN//divisor)):
    # output = tf.transpose(output_array)  # tf.transpose(output_array.stack())
    off_x, off_y, eos = sample(
        (enc_input, tf.expand_dims(output.stack(), axis=0)), 0, model)
    eos = tf.cast(eos, dtype=tf.float32)
    # current = tf.expand_dims(
    #     tf.concat((off_x, off_y, eos), axis=0)[:, 0], axis=0)
    current = tf.concat((off_x, off_y, eos), axis=0)  # [:, 0]
    output = output.write(i, current)

stroke = []
output = output.write(output.stack().shape[0], [0, 0, 1])
coords = offsets_to_coords(output.stack())
for point in coords:
    # output = tf.transpose(output_array)  # tf.transpose(output_array.stack())
    stroke.append((point[0], point[1]))
    if point[2] == 1:
        current_coords = list(zip(*stroke))
        plt.plot(current_coords[0], current_coords[1], 'k')
        stroke = []

# plt.plot(current_sample[:, 0], current_sample[:, 1])
plt.title(text)
plt.show()
