import tensorflow as tf
from data.stroke_utils import MAX_CHAR_LEN, MAX_STROKE_LEN, encode_ascii
from model.transformer import Transformer

model = Transformer()

checkpoint_path = "./checkpoints/train"
ckpt = tf.train.Checkpoint(transformer=model,
                           optimizer=model.optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
    print('Latest checkpoint restored!!')

text = "Dei mama"
enc_input = [encode_ascii(text)]
enc_input = tf.keras.preprocessing.sequence.pad_sequences(
    enc_input, maxlen=MAX_CHAR_LEN, padding='post')

start = tf.keras.preprocessing.sequence.pad_sequences(
    [[[0., 0., 1.]]], maxlen=MAX_STROKE_LEN, padding='post')

(coords, eos), attention_weights = model((enc_input, start))
print(coords)
