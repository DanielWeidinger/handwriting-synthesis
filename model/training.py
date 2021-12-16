import tensorflow as tf
from tensorflow.keras.optimizers.schedules import LearningRateSchedule


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


def get_optimizer(d_model):
    learning_rate = CustomSchedule(d_model)

    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                         epsilon=1e-9)
    return optimizer


def loss_func(label, logits, eos_loss_obj, coords_loss_obj):
    coords_label, eos_prob_label = (label[:, :, :2], label[:, :, 2:])
    coords_pred, eos_prob_pred = logits

    # End of sentence probability
    mask = tf.math.logical_not(tf.math.equal(eos_prob_label, 0))
    mask = tf.cast(mask, dtype=tf.int64)
    loss_eos = eos_loss_obj(
        eos_prob_label, eos_prob_pred, sample_weight=mask)

    # Coordinates (regression therefore MSE)
    loss_coords = coords_loss_obj(coords_label, coords_pred)

    return loss_coords, loss_eos


def accuracy_func(real, pred, threshold=0.8):
    real = real[:, :, 2:]
    pred = pred[:, :, 2:]
    print(real)
    accuracies = tf.equal(real, tf.where(pred > threshold))

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)

    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)
