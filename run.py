import tensorflow as tf
import numpy as np
import data.stroke_utils as su
from model.training import avg_error_distance, eos_accuracy


# real = tf.constant(
#     [[[1., 1., 0.], [2., 2., 0.], [3., 3., 1.]]], dtype=tf.float32)
real = tf.constant(
    [[[1, 1, 0], [1, 1, 0], [1, 1, 1]]], dtype=tf.float32)
pred = tf.constant(
    [[[0., 0., 0.9], [0., 0., 0.7], [0., 0., 0.9]]], dtype=tf.float32)

result = eos_accuracy(real, pred, [[1, 1, 0]])
print(result)
