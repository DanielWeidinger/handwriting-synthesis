from tensorflow.keras import Model, layers

from model.multi_head_attention import MultiHeadAttention


class Encoder(Model):
    def __init__(self, vocab_size, model_size, num_layers, h, pes):
        super(Encoder, self).__init__()
        self.model_size = model_size
        self.num_layers = num_layers
        self.h = h
        self.pes = pes

        # One Embedding layer
        self.embedding = layers.Embedding(vocab_size, model_size)

        # num_layers Multi-Head Attention and Normalization layers
        self.attention = [MultiHeadAttention(
            model_size, h) for _ in range(num_layers)]
        self.attention_norm = [layers.BatchNormalization()
                               for _ in range(num_layers)]

        # num_layers FFN and Normalization layers
        self.dense_1 = [layers.Dense(
            model_size * 4, activation='relu') for _ in range(num_layers)]
        self.dense_2 = [layers.Dense(model_size) for _ in range(num_layers)]
        self.ffn_norm = [layers.BatchNormalization()
                         for _ in range(num_layers)]

    def call(self, sequence, padding_mask):
        # padding_mask will have the same shape as the input sequence
        # padding_mask will be used in the Decoder too
        # so we need to create it outside the Encoder
        embed_out = self.embedding(sequence)
        embed_out += self.pes[:sequence.shape[1], :]

        sub_in = embed_out
        ffn_out = None

        for i in range(self.num_layers):
            sub_out = self.attention[i](sub_in, sub_in, padding_mask)
            sub_out = sub_in + sub_out
            sub_out = self.attention_norm[i](sub_out)

            ffn_in = sub_out

            ffn_out = self.dense_2[i](self.dense_1[i](ffn_in))
            ffn_out = ffn_in + ffn_out
            ffn_out = self.ffn_norm[i](ffn_out)

            sub_in = ffn_out

        return ffn_out
