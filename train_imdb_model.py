'''
Author: Brendan Reidy
'''


import os
import tempfile
import tensorflow as tf

import TransformerModel
import TransformerQuantization
import IMDB_Dataset
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
MAX_FEATURES = 10000
SEQUENCE_LENGTH = 250
EMBEDDING_DIM = 16
INTERMEDIATE_SIZE = 64
D_MODEL = 128
NUM_HEADS = 2
#strategy = tf.distribute.MirroredStrategy()#devices=["/gpu:0", "/gpu:1", "/gpu:2"])
strategy = tf.distribute.get_strategy()
BATCH_SIZE_PER_REPLICA = 16
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

EPOCHS = 1

def fetch_model(batch_size=None):
    '''
    fetch_model
    INPUT:
    OUTPUT
    '''
    with strategy.scope():
        inp = tf.keras.Input(shape=(SEQUENCE_LENGTH,), batch_size=batch_size, dtype=tf.float32, name="encoder_input", ragged=False)
        embedding = tf.keras.layers.Embedding(MAX_FEATURES+1, D_MODEL, input_length=SEQUENCE_LENGTH)(inp)
        #identity = embedding
        identity = TransformerModel.LinearLayer()(embedding)
        #out1 = TransformerModel.ScaledDotProduct(D_MODEL, int(D_MODEL/NUM_HEADS))(identity,identity,identity,None)
        #out1 = TransformerModel.BERTMultiHeadedAttention(NUM_HEADS, D_MODEL)(identity,identity,identity,None)
        out1 = TransformerModel.BertEncoder(NUM_HEADS, D_MODEL, INTERMEDIATE_SIZE, activation='relu')(identity, None)
        out1 = tf.keras.layers.Dropout(0.1)(out1)
        flat1 = tf.keras.layers.Flatten()(out1)
        output_layer =  tf.keras.layers.Dense(1, name='output_layer')(flat1)
    return tf.keras.Model(inputs=[inp], outputs=[output_layer], name="transformer")
  
train_ds, val_ds, test_ds = IMDB_Dataset.fetch_data(BATCH_SIZE)
model = fetch_model()
model.summary()

with strategy.scope():
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
      optimizer='adam',
      metrics=tf.metrics.BinaryAccuracy(threshold=0.0))

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS)

quant_aware_model = TransformerQuantization.QuantizeTransformer(model)
tf.keras.backend.clear_session()

quant_aware_model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer='adam',
              metrics=tf.metrics.BinaryAccuracy(threshold=0.0))
quant_aware_model.summary()

quant_aware_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS)

train_ds,val_ds,test_ds = IMDB_Dataset.fetch_data(1)
def representative_dataset():
    '''
    representative_dataset
    description:
      generates a representative dataset for quantized model
    '''
    for data in val_ds.take(400):
        yield [tf.dtypes.cast(tf.dtypes.cast(data[0], tf.uint8), tf.float32)]

quant_aware_model.input.set_shape((1,) + quant_aware_model.input.shape[1:])

float_converter = tf.lite.TFLiteConverter.from_keras_model(model)
float_tflite_model = float_converter.convert()

converter = tf.lite.TFLiteConverter.from_keras_model(quant_aware_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]


debugger = tf.lite.experimental.QuantizationDebugger(
    converter=converter, debug_dataset=representative_dataset)
debugger.run()
RESULTS_FILE = 'debugger_results/debugger_imdb_results.csv'
with open(RESULTS_FILE, 'w') as f:
    debugger.layer_statistics_dump(f)
#quantized_tflite_model = converter.convert()
quantized_tflite_model = debugger.get_nondebug_quantized_model()
"""
suspected_layers = []#["embedding", "encoder_input"]
debug_options = tf.lite.experimental.QuantizationDebugOptions(
    denylisted_nodes=suspected_layers)
debugger = tf.lite.experimental.QuantizationDebugger(
    converter=converter,
    debug_dataset=representative_dataset,
    debug_options=debug_options)
debugger.run()
RESULTS_FILE = 'debugger_results/debugger_imdb_results.csv'
with open(RESULTS_FILE, 'w') as f:
  debugger.layer_statistics_dump(f)
quantized_tflite_model = debugger.get_nondebug_quantized_model()
#quantized_tflite_model = converter.convert()
"""

_, quant_file = tempfile.mkstemp('.tflite')
_, float_file = tempfile.mkstemp('.tflite')
with open(float_file, 'wb') as f:
    f.write(float_tflite_model)

with open(quant_file, 'wb') as f:
    f.write(quantized_tflite_model)

print("Float model in Mb:", os.path.getsize(float_file) / float(2**20))
print("Quantized model in Mb:", os.path.getsize(quant_file) / float(2**20))

open("tflite_models/imdb_test_vehicle_int8.tflite", "wb").write(quantized_tflite_model)
open("tflite_models/imdb_test_vehicle_fp32.tflite", "wb").write(float_tflite_model)
print("Done")
