import TransformerModel
import tensorflow as tf
import numpy as np

partition_config = {
    "intermediate_partitions":1,
    "fc_out_partitions":1,
    "embedding_partitions":1,
    "use_conv":True
}

seq_len = 128
num_layers = 6
d_model = 512
num_heads = 8
intr_size = 2048

vocab_size = 100

def build_left_right():
    enc_mask = tf.keras.layers.Input(shape=(128), dtype=tf.float32, name="encoder_mask", ragged=False)
    enc_inp = tf.keras.layers.Input(shape=(128), dtype=tf.float32, name="input_word_ids", ragged=False)

    dec_mask = tf.keras.layers.Input(shape=(128), dtype=tf.float32, name="dcoder_mask", ragged=False)
    dec_inp = tf.keras.layers.Input(shape=(128), dtype=tf.float32, name="dec_inp", ragged=False)

    encoder = TransformerModel.Encoder(num_layers, num_heads, d_model, intr_size, vocab_size, seq_len,
        partition_config=partition_config)(enc_inp, enc_mask)
    decoder = TransformerModel.Decoder(num_layers, num_heads, d_model, intr_size, vocab_size, seq_len,
        partition_config=partition_config)(dec_inp, encoder, enc_mask, dec_mask)
    model = tf.keras.Model(inputs=[enc_inp, enc_mask, dec_inp, dec_mask], outputs=[decoder])
    return model

model = build_left_right()

np_data = np.random.randint(low=0,high=100,size=(1,128))
def representative_dataset():
    for data in np_data:
        yield [tf.dtypes.cast(data, tf.float32), tf.dtypes.cast(data, tf.float32), tf.dtypes.cast(data, tf.float32), tf.dtypes.cast(data, tf.float32)]

for input_idx in range(len(model.input)):
    model.input[input_idx].set_shape((1,) + model.input[input_idx].shape[1:])

model.summary()

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    
debugger = tf.lite.experimental.QuantizationDebugger(
    converter=converter, debug_dataset=representative_dataset)
debugger.run()
quantized_tflite_model = debugger.get_nondebug_quantized_model()

open("gpt_transformer.tflite", "wb").write(quantized_tflite_model)