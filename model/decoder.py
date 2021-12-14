import tensorflow as tf
from tensorflow.keras import Model

from model.multi_head_attention import MultiHeadAttention


class Decoder(Model):
    def __init__(self, seq_len, model_size, num_layers, h):
        super(Decoder, self).__init__()
        self.model_size = model_size
        self.num_layers = num_layers
        self.h = h
        self.seq_len = seq_len
        # self.pes = pes

        self.embedding = tf.keras.layers.Embedding(3, model_size)
        self.attention_bot = [MultiHeadAttention(
            model_size, h) for _ in range(num_layers)]
        self.attention_bot_norm = [
            tf.keras.layers.BatchNormalization() for _ in range(num_layers)]
        self.attention_mid = [MultiHeadAttention(
            model_size, h) for _ in range(num_layers)]
        self.attention_mid_norm = [
            tf.keras.layers.BatchNormalization() for _ in range(num_layers)]

        self.dense_1 = [tf.keras.layers.Dense(
            model_size * 4, activation='relu') for _ in range(num_layers)]
        self.dense_2 = [tf.keras.layers.Dense(
            model_size) for _ in range(num_layers)]
        self.ffn_norm = [tf.keras.layers.BatchNormalization()
                         for _ in range(num_layers)]

        self.coords_dense = tf.keras.layers.Dense(2)  # x and y
        self.eos_prob = tf.keras.layers.Dense(
            1, activation="sigmoid")  # x and y

    def call(self, sequence, encoder_output, padding_mask):
        # EMBEDDING AND POSITIONAL EMBEDDING
        # embed_out = embedding(sequence)
        # embed_out += pes[:sequence.shape[1], :]

        bot_sub_in = sequence  # embed_out
        ffn_out = None

        for i in range(self.num_layers):
            # BOTTOM MULTIHEAD SUB LAYER

            look_left_only_mask = tf.linalg.band_part(
                tf.ones((self.seq_len, self.seq_len)), -1, 0)
            bot_sub_out = self.attention_bot[i](
                bot_sub_in, bot_sub_in, look_left_only_mask)
            bot_sub_out = bot_sub_in + bot_sub_out
            bot_sub_out = self.attention_bot_norm[i](bot_sub_out)

            # MIDDLE MULTIHEAD SUB LAYER
            mid_sub_in = bot_sub_out

            mid_sub_out = self.attention_mid[i](
                mid_sub_in, encoder_output, padding_mask)
            mid_sub_out = mid_sub_out + mid_sub_in
            mid_sub_out = self.attention_mid_norm[i](mid_sub_out)

            # FFN
            ffn_in = mid_sub_out

            ffn_out = self.dense_2[i](self.dense_1[i](ffn_in))
            ffn_out = ffn_out + ffn_in
            ffn_out = self.ffn_norm[i](ffn_out)

            bot_sub_in = ffn_out

        coords_logits = self.coords_dense(ffn_out)
        eos_prob = self.eos_prob(ffn_out)

        return coords_logits, eos_prob
