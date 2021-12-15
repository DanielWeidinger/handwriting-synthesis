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
    def loss_func(targets, logits):
        crossentropy = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True)
        mask = tf.math.logical_not(tf.math.equal(targets, 0))
        mask = tf.cast(mask, dtype=tf.int64)
        loss = crossentropy(targets, logits, sample_weight=mask)

        return loss

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

            loss = Transformer.loss_func(target_seq_out, decoder_output)

        # variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        variables = self.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        return loss
