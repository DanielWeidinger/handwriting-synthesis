import tensorflow as tf
from tensorflow.keras import Model
import numpy as np
from data import stroke_utils as su
from model.decoder import Decoder
from model.encoder import Encoder

from model.utils import positional_embedding


class Transformer(Model):
    def __init__(self, model_size=128, num_layers=2, h=2):
        super(Transformer, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam()

        pes_alph = []
        for i in range(su.MAX_CHAR_LEN):
            pes_alph.append(positional_embedding(i, model_size))
        pes_alph = np.concatenate(pes_alph, axis=0)
        pes_alph = tf.constant(pes_alph, dtype=tf.float32)

        # TODO: try with pos enconding
        # pes_stroke = []
        # for i in range(su.MAX_STROKE_LEN):
        #     pes_stroke.append(positional_embedding(i, model_size))
        # pes_stroke = np.concatenate(pes_stroke, axis=0)
        # pes_stroke = tf.constant(pes_stroke, dtype=tf.float32)

        self.encoder = Encoder(
            su.alphabet_len, model_size, num_layers, h, pes_alph)
        self.decoder = Decoder(su.MAX_STROKE_LEN, model_size, num_layers, h)

    def __call__(self, test):
        pass

    @staticmethod
    def loss_func(label, logits):
        coords_label, eos_prob_label = (label[:, :, :2], label[:, :, 2:])
        coords_pred, eos_prob_pred = logits

        # End of sentence probability
        crossentropy = tf.keras.losses.BinaryCrossentropy()
        mask = tf.math.logical_not(tf.math.equal(eos_prob_label, 0))
        mask = tf.cast(mask, dtype=tf.int64)
        loss_eos = crossentropy(
            eos_prob_label, eos_prob_pred, sample_weight=mask)

        # Coordinates (regression therefore MSE)
        mse = tf.keras.losses.MeanSquaredError()
        loss_coords = mse(coords_label, coords_pred)

        return loss_coords, loss_eos

    @tf.function
    def train_step(self, source_seq, target_seq_in, target_seq_out):
        with tf.GradientTape() as tape:
            padding_seq = tf.squeeze(source_seq[:, :, :1])
            # TODO: fix so that /x00 is not in the alphabet but rather the marking of padding
            padding_mask = 1 - \
                tf.cast(tf.equal(padding_seq, 0), dtype=tf.float32)

            # Manually add two more dimentions
            # so that the mask's shape becomes (batch_size, 1, 1, seq_len)
            padding_mask = tf.expand_dims(padding_mask, axis=1)
            padding_mask = tf.expand_dims(padding_mask, axis=1)

            encoder_output = self.encoder(source_seq, padding_mask)

            decoder_output = self.decoder(
                target_seq_in, encoder_output, padding_mask)

            loss_coords, loss_eos = Transformer.loss_func(
                target_seq_out, decoder_output)
            loss = loss_coords + loss_eos

        # variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        variables = self.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        return loss, loss_coords, loss_eos
