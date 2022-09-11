import matplotlib.pyplot as plt
import os
import re
import tempfile
import shutil
import string
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses
import tensorflow_model_optimization as tfmot
import numpy as np
import TransformerModel
import TransformerQuantization
import IMDB_Dataset
from tensorboard import main as tb
import threading
import os
import MaskUtils
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

tensorBoardPath = "/home/brendan/tpu_transformer/tensorboard/"

def launchTensorBoard():
    import os
    os.system('tensorboard --logdir=' + tensorBoardPath)
    return

t = threading.Thread(target=launchTensorBoard, args=([]))
t.start()

max_features = 10000
sequence_length = 250
embedding_dim = 16
intermediate_size = 64
d_model = 128
num_heads = 2
#strategy = tf.distribute.MirroredStrategy()#devices=["/gpu:0", "/gpu:1", "/gpu:2"])
strategy = tf.distribute.get_strategy()
BATCH_SIZE_PER_REPLICA = 16
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

def fetchRawModel(batch_size=None):
  with strategy.scope():
    x = tf.keras.Input(shape=(sequence_length,), batch_size=batch_size, dtype=tf.float32, name="encoder_input", ragged=False)
    embedding = tf.keras.layers.Embedding(max_features+1, d_model, input_length=sequence_length)(x)
    #identity = embedding
    identity = TransformerModel.LinearLayer()(embedding)
    #out1 = TransformerModel.ScaledDotProduct(d_model, int(d_model/num_heads))(identity,identity,identity,None)
    #out1 = TransformerModel.BERTMultiHeadedAttention(num_heads, d_model)(identity,identity,identity,None)
    out1 = TransformerModel.BertEncoder(num_heads, d_model, intermediate_size, activation='relu')(identity, None)
    flat1 = tf.keras.layers.Flatten()(out1)
    output_layer =  tf.keras.layers.Dense(1, name='output_layer')(flat1)
    model = tf.keras.Model(inputs=[x], outputs=[output_layer], name="transformer")
  return model

def compileQuantizedModel(tf_quantModel):
  with strategy.scope():
    rawModel = fetchRawModel(batch_size=1)
    rawModel.compile(loss=losses.BinaryCrossentropy(from_logits=True),
                optimizer='adam',
                metrics=tf.metrics.BinaryAccuracy(threshold=0.0))
    quant_aware_model = TransformerQuantization.QuantizeTransformer(rawModel)
    quant_aware_model.compile(loss=losses.BinaryCrossentropy(from_logits=True),
                  optimizer='adam',
                  metrics=tf.metrics.BinaryAccuracy(threshold=0.0))
    quant_aware_model.set_weights(tf_quantModel.get_weights())
  return quant_aware_model
  
train_ds,val_ds,test_ds = IMDB_Dataset.fetch_data(BATCH_SIZE)
epochs = 1
model = fetchRawModel()
model.summary()

with strategy.scope():
  model.compile(loss=losses.BinaryCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=tf.metrics.BinaryAccuracy(threshold=0.0))

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorBoardPath, histogram_freq=1)
history = model.fit(
    train_ds,
    #validation_data=val_ds,
    epochs=epochs,
    callbacks=[tensorboard_callback])

float_converter = tf.lite.TFLiteConverter.from_keras_model(model)
float_tflite_model = float_converter.convert()
_, float_file = tempfile.mkstemp('.tflite')
with open(float_file, 'wb') as f:
  f.write(float_tflite_model)
floatSize = os.path.getsize(float_file) / float(2**20)
print("FP size",floatSize)
#open("models/imdb_fp32_d_model512.tflite", "wb").write(float_tflite_model)
#print("Done saving")
#"""
quant_aware_model = TransformerQuantization.QuantizeTransformer(model)
tf.keras.backend.clear_session()

quant_aware_model.compile(loss=losses.BinaryCrossentropy(from_logits=True),
              optimizer='adam',
              metrics=tf.metrics.BinaryAccuracy(threshold=0.0))
quant_aware_model.summary()

quant_aware_model.fit(
    train_ds,
    #validation_data=val_ds,
    epochs=1,
    callbacks=[tensorboard_callback])

train_ds,val_ds,test_ds = IMDB_Dataset.fetch_data(1)
def representative_dataset():
  for data in val_ds.take(1):
    yield [tf.dtypes.cast(tf.dtypes.cast(data[0], tf.uint8), tf.float32)]
#quant_aware_model = compileQuantizedModel(quant_aware_model)
quant_aware_model.input.set_shape((1,) + quant_aware_model.input.shape[1:])
print("new shape",quant_aware_model.input.shape)

converter = tf.lite.TFLiteConverter.from_keras_model(quant_aware_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
#converter.inference_input_type = tf.int8  # or tf.uint8
#converter.inference_output_type = tf.int8  # or tf.uint8
#quantized_tflite_model = converter.convert()
suspected_layers = ["embedding", "encoder_input"]
"""
debug_options = tf.lite.experimental.QuantizationDebugOptions(
    denylisted_nodes=suspected_layers)
debugger = tf.lite.experimental.QuantizationDebugger(
    converter=converter,
    debug_dataset=representative_dataset,
    debug_options=debug_options)
"""

#quantized_tflite_model = debugger.get_nondebug_quantized_model()
quantized_tflite_model = converter.convert()

_, quant_file = tempfile.mkstemp('.tflite')

with open(quant_file, 'wb') as f:
  f.write(quantized_tflite_model)

print("Float model in Mb:", floatSize)
print("Quantized model in Mb:", os.path.getsize(quant_file) / float(2**20))

open("tflite_models/imdb_test_vehicle_int8.tflite", "wb").write(quantized_tflite_model)
open("tflite_models/imdb_test_vehicle_fp32.tflite", "wb").write(float_tflite_model)

print("Done")
#"""