import tensorflow as tf
import numpy as np
import data.stroke_utils as su
from model.training import accuracy_func


real = tf.constant(
    [[[0, 0, 0], [0, 0, 0], [0, 0, 1]]], dtype=tf.int64)
pred = tf.constant(
    [[[0., 0., 0.3], [0., 0., 0.7], [0., 0., 0.9]]], dtype=tf.float32)

result = accuracy_func(real, pred)
result
