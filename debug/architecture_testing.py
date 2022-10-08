import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"]="-1"

if len(sys.argv) < 3:
    print("Usage:", sys.argv[0], " <d_model> <emb_size>")
    exit()

import tensorflow as tf
import numpy as np

D_MODEL = int(sys.argv[1])
EMB_SIZE = int(sys.argv[2])

tf.random.set_seed(42)
tf.keras.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism()
tf.config.experimental.enable_op_determinism()
np.random.seed(42)
initializer = tf.keras.initializers.GlorotUniform(seed=42)

VOCAB_SIZE = 10000
SEQUENCE_LENGTH = 128

LOW=0
HIGH=100

n_samples = 1

inp = tf.keras.Input(shape=(SEQUENCE_LENGTH,), dtype=tf.float32, name="encoder_input", ragged=False)
embedding = tf.keras.layers.Embedding(VOCAB_SIZE+1, EMB_SIZE, input_length=SEQUENCE_LENGTH, embeddings_initializer=initializer)(inp)
x = tf.keras.layers.Dense(D_MODEL, kernel_initializer=initializer, bias_initializer=initializer)(embedding)
x = x[:,0,:]
out = tf.keras.layers.Dense(2, kernel_initializer=initializer, bias_initializer=initializer)(x)

model = tf.keras.Model(inputs=[inp], outputs=[out])

model.summary()

np_data = np.random.uniform(low=LOW,high=HIGH,size=(n_samples, SEQUENCE_LENGTH))

print(np_data.shape)
def representative_dataset():
  for data in np_data:
    yield [tf.dtypes.cast(data, tf.float32)]


model.input.set_shape((1,) + model.input.shape[1:])

import tempfile

float_converter = tf.lite.TFLiteConverter.from_keras_model(model)
float_tflite_model = float_converter.convert()

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
quantized_tflite_model = converter.convert()

_, quant_file = tempfile.mkstemp('.tflite')
_, float_file = tempfile.mkstemp('.tflite')
with open(float_file, 'wb') as f:
  f.write(float_tflite_model)

with open(quant_file, 'wb') as f:
  f.write(quantized_tflite_model)

print("Float model in Mb:", os.path.getsize(float_file) / float(2**20))
print("Quantized model in Mb:", os.path.getsize(quant_file) / float(2**20))
"""
passed = False

while not passed:
    #try:
    debugger = tf.lite.experimental.QuantizationDebugger(
        converter=converter, debug_dataset=representative_dataset)
    debugger.run()
    quantized_tflite_model = debugger.get_nondebug_quantized_model()
    passed=True
    #except:
    #    passed=False
"""
open("tflite_models/arch_test_int8.tflite", "wb").write(quantized_tflite_model)