import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.keras.losses import Loss
from tensorflow.python.keras.utils import losses_utils


class CustomSchedule(LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


class NegativeLogLikelihood(Loss):
    def __init__(self, reduction=losses_utils.ReductionV2.AUTO, name="negative_log_loss", epsilon=1e-8):
        super().__init__(reduction=reduction, name=name)
        self.epsilon = epsilon

    def call(self, y_true, y_pred):
        x, y, eos_prob = y_true
        mixture_weights, stddev1, stddev2, mean1, mean2, correl, eos_prob_pred = y_pred

        norm = 1.0 / (2*np.pi*stddev1*stddev2 * tf.sqrt(1 - tf.square(correl)))
        Z = tf.math.square((x - mean1) / (stddev1)) + \
            tf.math.square((y - mean2) / (stddev2)) - \
            2*correl*(x - mean1)*(y - mean2) / (stddev1*stddev2)

        exp = -1.0*Z / (2*(1 - tf.math.square(correl)))
        gaussian_likelihoods = tf.exp(exp) * norm
        gmm_likelihood = tf.reduce_sum(
            mixture_weights * gaussian_likelihoods, 2)
        gmm_likelihood = tf.clip_by_value(gmm_likelihood, self.epsilon, np.inf)

        bernoulli_likelihood = tf.squeeze(tf.where(
            tf.equal(tf.ones_like(eos_prob), eos_prob), eos_prob_pred, eos_prob_pred))
        bernoulli_likelihood = tf.clip_by_value(
            bernoulli_likelihood, self.epsilon, 1)

        nll = -(tf.math.log(gmm_likelihood) +
                tf.math.log(bernoulli_likelihood))

        return nll

        # Z = tf.math.square((x - mean1) / stddev1) + tf.math.square((y - mean2) / stddev2) \
        #     - (2 * correl * (x - mean1) * (y - mean2) / (stddev1 * stddev2))

        # bivarian_gaussian = tf.math.exp(-Z / (2 * (1 - tf.math.square(correl)))) \
        #     / (2 * np.pi * stddev1 * stddev2 * tf.math.sqrt(1 - tf.math.square(correl)))
        # bivarian_gaussian *= mixture_weights
        # gaussian_loss = tf.math.log(tf.reduce_sum(
        #     bivarian_gaussian, axis=-1, keepdims=True)+self.epsilon)

        # bernoulli_loss = tf.where(tf.math.equal(tf.ones_like(
        #     eos_prob), eos_prob), eos_prob_pred, 1 - eos_prob_pred)
        # bernoulli_loss = tf.math.log(bernoulli_loss+self.epsilon)

        # negative_log_loss = -gaussian_loss - bernoulli_loss
        # return tf.math.reduce_sum(negative_log_loss, axis=-1)
        # return tf.reduce_mean(tf.math.reduce_sum(negative_log_loss, axis=1))


def get_optimizer(d_model):
    learning_rate = CustomSchedule(d_model)

    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                         epsilon=1e-9)
    return optimizer


def loss_func(label, pred, mask, loss_obj):
    y_true = (label[:, :, 0:1], label[:, :, 1:2], label[:, :, 2:])
    loss = loss_obj(y_true, pred, sample_weight=mask)

    return loss


def eos_accuracy(real, pred, mask):
    real = tf.squeeze(tf.cast(real[:, :, 2:], dtype=tf.int32))
    _, _, eos_pred = pred

    accuracy = tf.cast(tf.reduce_sum(
        tf.cast(real == eos_pred, dtype=tf.int32)), dtype=tf.float64)/tf.reduce_sum(mask)
    return accuracy


def mean_sqared_error(real, pred, mask):
    real = real[:, :, :2]
    x_pred, y_pred, _ = pred

    squared_error = (real[:, :, 0] - x_pred)**2 + (real[:, :, 1] - y_pred)**2
    squared_error = squared_error[mask == 1]

    mse = tf.reduce_sum(squared_error, axis=-1)/squared_error.shape[0]

    return mse


def avg_error_distance(real, pred):
    real = real[:, :, :2]
    pred = pred[:, :, :2]
    vectors = real - pred
    vec_length = tf.math.sqrt(tf.math.add(tf.math.square(
        vectors[:, :, :1]), tf.math.square(vectors[:, :, 1:2])))

    return tf.reduce_sum(vec_length)/vec_length.shape[1]
