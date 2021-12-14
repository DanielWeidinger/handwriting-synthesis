import tensorflow as tf
from tensorflow.keras import Model


class MultiHeadAttention(Model):
    def __init__(self, model_size, h):
        super(MultiHeadAttention, self).__init__()
        self.key_size = model_size // h
        self.h = h
        # [tf.keras.layers.Dense(key_size) for _ in range(h)]
        self.wq = tf.keras.layers.Dense(model_size)
        # [tf.keras.layers.Dense(key_size) for _ in range(h)]
        self.wk = tf.keras.layers.Dense(model_size)
        # [tf.keras.layers.Dense(value_size) for _ in range(h)]
        self.wv = tf.keras.layers.Dense(model_size)
        self.wo = tf.keras.layers.Dense(model_size)

    def call(self, decoder_output, encoder_output, mask=None):
        query = self.wq(decoder_output)
        key = self.wk(encoder_output)
        value = self.wv(encoder_output)

        # Split for multihead attention
        batch_size = query.shape[0]
        query = tf.reshape(query, [batch_size, -1, self.h, self.key_size])
        query = tf.transpose(query, [0, 2, 1, 3])
        key = tf.reshape(key, [batch_size, -1, self.h, self.key_size])
        key = tf.transpose(key, [0, 2, 1, 3])
        value = tf.reshape(value, [batch_size, -1, self.h, self.key_size])
        value = tf.transpose(value, [0, 2, 1, 3])

        score = tf.matmul(query, key, transpose_b=True)
        score /= tf.math.sqrt(tf.dtypes.cast(self.key_size, dtype=tf.float32))

        if mask is not None:
            score *= mask
            score = tf.where(tf.equal(score, 0),
                             tf.ones_like(score) * -1e9, score)

        alignment = tf.nn.softmax(score, axis=-1)
        context = tf.matmul(alignment, value)
        context = tf.transpose(context, [0, 2, 1, 3])
        context = tf.reshape(context, [batch_size, -1, self.key_size * self.h])

        heads = self.wo(context)
        # heads has shape (batch, decoder_len, model_size)
        return heads
