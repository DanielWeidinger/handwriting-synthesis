import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from data import stroke_utils as su
from model.decoder import Decoder
from model.encoder import Encoder
from model.training import get_optimizer, loss_func, NegativeLogLikelihood
from model.utils import create_look_ahead_mask, create_padding_mask


class Transformer(Model):
    def __init__(self, num_layers=4, d_model=128, num_heads=8, dff=512, input_vocab_size=su.alphabet_len,
                 pe_input=su.MAX_CHAR_LEN, pe_target=su.MAX_STROKE_LEN, rate=0.1, num_mixtures=10):
        super().__init__()

        self.optimizer = get_optimizer(d_model)

        self.nll = NegativeLogLikelihood()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                               input_vocab_size, pe_input, rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                               pe_target, rate)

        self.num_mixtures = num_mixtures
        # self.final_layer = tf.keras.layers.Dense(target_vocab_size)
        # TODO: try without Dense on decoder output
        # num_mixtures*(1x pis(distribution weights), 2x sigmas, 2x mus, 1x rho(pearson'correlation) ), 1x eos(bernulli)
        self.outputs = tf.keras.layers.Dense(6*num_mixtures+1)

    def call(self, inputs, training):
        # Keras models prefer if you pass all your inputs in the first argument
        inp, tar = inputs

        enc_padding_mask, look_ahead_mask, dec_padding_mask = self.create_masks(
            inp, tar)

        # (batch_size, inp_seq_len, d_model)
        enc_output = self.encoder(inp, training, enc_padding_mask)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask)

        # (batch_size, tar_seq_len, target_vocab_size)
        # final_output = self.final_layer(dec_output)

        # Mixed Density network
        Z = self.outputs(dec_output)
        parameter = self.parameterize_distributions(Z)

        return parameter, attention_weights

    def parameterize_distributions(self, Z, eps=1e-8, sigma_eps=1e-4):
        pis, sigmas, mus, rhos, es = tf.split(
            Z,
            [1*self.num_mixtures, 2*self.num_mixtures, 2 *
                self.num_mixtures, 1*self.num_mixtures, 1],
            axis=-1
        )

        pis = tf.nn.softmax(pis, axis=-1)
        sigmas = tf.clip_by_value(tf.exp(sigmas), sigma_eps, np.inf)
        rhos = tf.clip_by_value(tf.tanh(rhos), eps - 1.0, 1.0 - eps)
        es = tf.clip_by_value(tf.nn.sigmoid(es), eps, 1.0 - eps)

        sigma1, sigma2 = tf.split(
            sigmas, [1*self.num_mixtures, 1*self.num_mixtures], axis=-1)
        mu1, mu2 = tf.split(
            mus, [1*self.num_mixtures, 1*self.num_mixtures], axis=-1)

        return pis, sigma1, sigma2, mu1, mu2, rhos, es

    def create_masks(self, inp, tar):
        # Encoder padding mask
        enc_padding_mask = create_padding_mask(inp)

        # Used in the 2nd attention block in the decoder.
        # This padding mask is used to mask the encoder outputs.
        dec_padding_mask = create_padding_mask(inp)

        # Used in the 1st attention block in the decoder.
        # It is used to pad and mask future tokens in the input received by
        # the decoder.
        look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
        padding_seq = tf.math.reduce_sum(tar, axis=-1)
        dec_target_padding_mask = create_padding_mask(padding_seq)
        look_ahead_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        return enc_padding_mask, look_ahead_mask, dec_padding_mask

    # The @tf.function trace-compiles train_step into a TF graph for faster
    # execution. The function specializes to the precise shape of the argument
    # tensors. To avoid re-tracing due to the variable sequence lengths or variable
    # batch sizes (the last batch is smaller), use input_signature to specify
    # more generic shapes.
    # (input_signature=[tf.TensorSpec(shape=(None, None), dtype=tf.int64), tf.TensorSpec(shape=(None, None), dtype=tf.int64), tf.TensorSpec(shape=(None, None), dtype=tf.int64)])
    @tf.function()
    def train_step(self, inp, tar_inp, tar_out, tar_mask):
        with tf.GradientTape() as tape:
            predictions, _ = self([inp, tar_inp],
                                  training=True)
            loss = loss_func(tar_out, predictions, tar_mask, self.nll)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.trainable_variables))

        return predictions, loss
