import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"]="-1"

if len(sys.argv) < 3:
    print("Usage:", sys.argv[0], " <d_model> <emb_size>")
    exit()

import tensorflow as tf
import numpy as np
import TransformerModel

SWAP_DICT = {
  3072:12,
  2048:8,
  1024:4,
  768:6
}

D_MODEL = int(sys.argv[1])
EMB_SIZE = int(sys.argv[2])

tf.random.set_seed(42)
tf.keras.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism()
np.random.seed(42)
initializer = tf.keras.initializers.GlorotUniform(seed=42)

VOCAB_SIZE = 10000
SEQUENCE_LENGTH = 128

LOW=0
HIGH=1


class PartitionLayer(tf.keras.layers.Layer):
    def __init__(self, output_size, input_size, num_layers=1, partition_output=True, rank=2, name=None):
        super(PartitionLayer, self).__init__(name=name)
        if partition_output:
          assert output_size%num_layers==0
        else:
          assert input_size%num_layers==0
        self.num_layers = num_layers
        self.output_size = output_size
        self.rank = rank
        self.input_size=input_size
        self.partition_output = partition_output
        self.fcs = []
        for i in range(num_layers):
          if partition_output:
            self.fcs.append(TransformerModel.Dense(int(output_size/num_layers), inp_size=input_size, use_conv=True))
          else:
            self.fcs.append(TransformerModel.Dense(output_size, inp_size=int(input_size/num_layers), use_conv=True))

    def build(self, input_shape):
      pass

    def get_config(self):
        return {
            'name':self.name}

    def call(self, x):
      if self.partition_output:
        outputs = []
        for i in range(self.num_layers):
          outputs.append(self.fcs[i](x))
        x = outputs[0]
        if self.num_layers>1:
            x = tf.keras.layers.concatenate(outputs)
        return x
      rank = self.rank
      partition_size = int(self.input_size/self.num_layers)
      if rank==2:
        output = self.fcs[0](x[:,0:partition_size])
        for i in range(1,self.num_layers):
          output+=self.fcs[i](x[:,partition_size*(i):partition_size*(i+1)])
        return output
      elif rank==3:
        output = self.fcs[0](x[:,:,0:partition_size])
        for i in range(1,self.num_layers):
          output+=self.fcs[i](x[:,:,partition_size*(i):partition_size*(i+1)])
        return output
      return x

class PartitionEmbedding(tf.keras.layers.Layer):
  def __init__(self, vocab_Size, emb_size, input_length=None, n_partitions=1, name=None):
      super(PartitionEmbedding, self).__init__(name=name)
      self.vocab_size = vocab_Size
      self.emb_size = emb_size
      self.input_length = input_length
      self.n_partitions = n_partitions
      self.partition_size = int(self.emb_size/n_partitions)
      assert self.emb_size % n_partitions == 0
      self.embeddings = [tf.keras.layers.Embedding(vocab_Size, self.partition_size, input_length=input_length) for i in range(n_partitions)]

  def call(self, x):
      outputs = []
      for embedding in self.embeddings:
          outputs.append(embedding(x))
      x = outputs[0]
      if self.n_partitions>1:
          x = tf.keras.layers.concatenate(outputs)
      return x

  
n_layers1 = 1
partition_output=True      
if EMB_SIZE in SWAP_DICT.keys():
  partition_output=False
  n_layers1 = SWAP_DICT[EMB_SIZE]
  print("ERERERARASSRFAS")


n_samples = 1

inp = tf.keras.Input(shape=(SEQUENCE_LENGTH,), dtype=tf.float32, name="encoder_input", ragged=False)
embedding = PartitionEmbedding(VOCAB_SIZE+1, 768, n_partitions=2)(inp)
x = TransformerModel.Dense(D_MODEL, inp_size=768, use_conv=True)(embedding)
x = TransformerModel.Dense(EMB_SIZE, inp_size=D_MODEL, use_conv=True)(x)
x = x[:,0,:]
out = TransformerModel.Dense(2, inp_size=EMB_SIZE, use_conv=True)(x)
#out = TransformerModel.Dense(2, inp_size=D_MODEL, use_conv=True)(x)

model = tf.keras.Model(inputs=[inp], outputs=[out])

model.summary()

for w in model.weights:
  if "kernel" in w.name:
    print(w.name, w.shape)
  elif "embedding" in w.name:
    print(w.name, w.shape, tf.math.reduce_max(w), tf.math.reduce_min(w), tf.math.reduce_mean(w))

np_data = np.random.uniform(low=LOW,high=HIGH,size=(n_samples, SEQUENCE_LENGTH))

print(model.predict(np_data))

print(np_data.shape)
def representative_dataset():
  for data in np_data:
    yield [tf.dtypes.cast(np.reshape(data, (1,SEQUENCE_LENGTH)), tf.float32)]


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

open("tflite_models/arch_test_int8_2.tflite", "wb").write(quantized_tflite_model)