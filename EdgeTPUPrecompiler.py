import ConvertModel
import TransformerModel
import numpy as np
import tensorflow as tf

def update_dict(original, changes):
    for key in changes.keys():
        if key in original.keys():
            original[key] = changes[key]
    return original

def partition_model(model, intermediate_partitions=1):
    for layer in model.layers:
        if isinstance(layer, TransformerModel.BertEncoder) or isinstance(layer, TransformerModel.BERT) :
            layer.set_partitions(intermediate_partitions)
    model = tf.keras.Model(inputs=model.input, outputs=model.output)
    return model

def precompile_model(model):
    #TODO standardize pre-compilation process for model
    pass
"""
    #print(new_model)
MAX_FEATURES = 10000
SEQUENCE_LENGTH = 250
EMBEDDING_DIM = 16
INTERMEDIATE_SIZE = 64
D_MODEL = 128
NUM_HEADS = 2
val_x = np.loadtxt("data/IMDB_val_x")
val_y = np.loadtxt("data/IMDB_val_y")
print(val_x.shape)

inp = tf.keras.Input(shape=(SEQUENCE_LENGTH,), dtype=tf.float32, name="encoder_input", ragged=False)
embedding = tf.keras.layers.Embedding(MAX_FEATURES+1, D_MODEL, input_length=SEQUENCE_LENGTH)(inp)
#identity = embedding
identity = TransformerModel.LinearLayer()(embedding)
#out1 = TransformerModel.ScaledDotProduct(D_MODEL, int(D_MODEL/NUM_HEADS))(identity,identity,identity,None)
#out1 = TransformerModel.BERTMultiHeadedAttention(NUM_HEADS, D_MODEL)(identity,identity,identity,None)
out1 = TransformerModel.BertEncoder(NUM_HEADS, D_MODEL, INTERMEDIATE_SIZE, activation='relu')(identity, None)
out1 = tf.keras.layers.Dropout(0.1)(out1)
flat1 = tf.keras.layers.Flatten()(out1)
output_layer =  tf.keras.layers.Dense(1, name='output_layer')(flat1)
model = tf.keras.Model(inputs=[inp], outputs=[output_layer], name="transformer")
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
      optimizer='adam',
      metrics=tf.metrics.BinaryAccuracy(threshold=0.0))

history = model.fit(
    val_x, val_y,
    epochs=2)

model.evaluate(val_x, val_y)
model = partition_model(model, intermediate_partitions=2)
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
      optimizer='adam',
      metrics=tf.metrics.BinaryAccuracy(threshold=0.0))
model.evaluate(val_x, val_y)
"""