import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from data.stroke_utils import MAX_CHAR_LEN, MAX_STROKE_LEN, encode_ascii, offsets_to_coords
from model.transformer import Transformer

model = Transformer()


def sample(inputs, seed):
    (mixture_weight, stddev1, stddev2, mean1, mean2, correl, end_stroke), _ = model(
        inputs)
    # = model.parameterize_distributions(
    #     outputs)

    # sample for MDN index from mixture weights
    mixture_dist = tfp.distributions.Categorical(
        probs=mixture_weight[0, 0])
    mixture_idx = mixture_dist.sample(seed=seed)

    # retrieve correct distribution values from mixture
    mean1 = tf.gather(mean1, mixture_idx, axis=-1)
    mean2 = tf.gather(mean2, mixture_idx, axis=-1)
    stddev1 = tf.gather(stddev1, mixture_idx, axis=-1)
    stddev2 = tf.gather(stddev2, mixture_idx, axis=-1)
    correl = tf.gather(correl, mixture_idx, axis=-1)

    # sample for x, y offsets
    cov_matrix = [[stddev1 * stddev1, correl * stddev1 * stddev2],
                  [correl * stddev1 * stddev2, stddev2 * stddev2]]
    bivariate_gaussian_dist = tfp.distributions.MultivariateNormalDiag(
        loc=[mean1, mean2], scale_diag=cov_matrix)
    bivariate_sample = bivariate_gaussian_dist.sample(seed=seed)
    x, y = bivariate_sample[0, 0], bivariate_sample[1, 1]

    # sample for end of stroke
    bernoulli = tfp.distributions.Bernoulli(probs=end_stroke)
    end_cur_stroke = bernoulli.sample(seed=seed)
    return [x, y, end_cur_stroke]


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

output = np.zeros((1, MAX_STROKE_LEN, 3), dtype=np.float32)
output[0, 0] = [0, 0, 1]
output[0, -1] = [0, 0, 1]

divisor = 1
for i in tqdm(tf.range(1, MAX_STROKE_LEN//divisor)):
    # output = tf.transpose(output_array)  # tf.transpose(output_array.stack())
    off_x, off_y, eos = sample((enc_input, output), 0)
    eos = tf.cast(tf.squeeze(eos, axis=-1), dtype=tf.float32)
    current = tf.concat((off_x, off_y, eos), axis=0)[:, 0]
    output[0, i] = current

stroke = []
coords = offsets_to_coords(output[0])
for point in coords:
    # output = tf.transpose(output_array)  # tf.transpose(output_array.stack())
    stroke.append((point[0], point[1]))
    if point[2] == 1:
        coords = list(zip(*stroke))
        plt.plot(coords[0], coords[1], 'k')
        stroke = []

# plt.plot(current_sample[:, 0], current_sample[:, 1])
plt.title(text)
plt.show()
