import tensorflow as tf
import numpy as np
import tensorflow_model_optimization as tfmot
from keras import activations
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.ops import nn_ops
import functools
import math
import json
import TransformerModel

class DynamicLayerNormalization(tf.keras.layers.Layer):
    """
    Tensorflow implmentation of integer-only layer norm
    https://github.com/kssteven418/I-BERT
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/ibert/quant_modules.py

    """
    
    def __init__(self, epsilon=1e-6, name=None):
        super(DynamicLayerNormalization, self).__init__(name=name)
        self.epsilon = epsilon
        self.outter_dim = None
        self.middle_dim = None
        self.alpha = 1
        self.epochs = 0
        self.transition_epoch = 60
        self.dropout = tf.keras.layers.Dropout(0.1)

    def get_config(self):
        return {
            'epsilon': self.epsilon,
            'name':self.name
            }

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        self.middle_dim = input_shape[-2]
        self.outter_dim = input_shape[-1]
        self.weight = self.add_weight("gamma",shape=[input_shape[-1]],
                initializer='random_normal',
                trainable=True)

        #beta
        self.bias = self.add_weight("beta",shape=[input_shape[-1]],
                initializer='random_normal',
                trainable=True)

    def call(self, x):
        mean, var = tf.nn.moments(x, -1, keepdims=True)
        #xsum = tf.math.reduce_sum(x, axis=-1, keepdims=True)
        #mean = xsum/self.outter_dim
        #x = y*tf.math.rsqrt(self.epsilon + var)
        y = x - mean
        xmin = tf.math.reduce_min(y, axis=-2, keepdims=True)
        xmax = tf.math.reduce_max(y, axis=-2, keepdims=True)

        scale_adjustment = 1 / (math.sqrt(2*np.log(self.outter_dim)))
        x = (y)/((xmax-xmin)*scale_adjustment)
        x = x * self.weight + self.bias
        return x

#
# Author: Brendan Reidy
# Email: bcreidy@email.sc.edu
# Date created: May 5 2020
# Last Modified: Sep 11 2022
#

import tensorflow as tf
import numpy as np
import tensorflow_model_optimization as tfmot
from keras import activations
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.ops import nn_ops
import functools
import math
import json

LastValueQuantizer = tfmot.quantization.keras.quantizers.LastValueQuantizer
MovingAverageQuantizer = tfmot.quantization.keras.quantizers.MovingAverageQuantizer

DEFAULT_PARTITION_CONFIG = {
    "intermediate_partitions":1,
    "fc_out_partitions":1,
    "embedding_partitions":1,
    "use_conv":False
}

def deserialize_partition_config(partition_config):
    if partition_config==None:
        partition_config=DEFAULT_PARTITION_CONFIG
    if isinstance(partition_config, str):
        partition_config=json.loads(partition_config)
    return partition_config

def relu(x):
    y = tf.constant([0.])
    return tf.math.maximum(x, y)

erf_a = -0.2888
erf_b = -1.769
erf_c = 1

epsilon = 1e-3

k_factor = 10
k_factor_sqr = k_factor**2

def int_rsqrt(x):
    #num = (x**4 + 112*(x**3) + 1120*(x**2) + 1792*x + 256)
    #denom = (16*(x**3) + 448*(x**2) + 1792*x + 1024)
    #return (num / (denom + epsilon))

    a = tf.math.rsqrt(x + epsilon)
    b = tf.zeros_like(a)

    #NaNs in a replaced with values in b
    c = tf.where(tf.math.is_inf(a), b, a)
    return c

#abs function is not supported on edge tpu
def tf_abs(x):
    return tf_sign(x)*x

def tf_sign(x):
    return x*int_rsqrt(x**2)
    #denominator = 1 + tf.math.exp(-2000*x)
    #return (2/(denominator + epsilon)) - 1
    #return (2/((1 + tf.math.exp(-2000*x))+epsilon)) - 1
    #return tf.tanh(x*1e3)

def approx_erf(x):
    return tf_sign(x)*(erf_a * (tf.math.minimum(tf_abs(x), -erf_b)+erf_b)**2 + erf_c)

def approx_gelu(x):
    #return x*0.5*(1+approx_erf(x/math.sqrt(2)))
    return x*tf.math.sigmoid(1.702*x)


#sign function is not supported in tflite
#return tf.tanh(x*1e3)
#return tf_sign(x)*x
def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
  
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

class ApproxGelu():
    def __init__(self):
        self.sigmoid = activations.get("sigmoid")

    def __call__(self, x):
        return x*self.sigmoid(x*1.702)

    def get_activations(self):
        return self.sigmoid

    def set_activations(self, act):
        self.sigmoid = act


class CustomLayer(tf.keras.layers.Layer):
    def __init__(self, k, name=None):
        super(CustomLayer, self).__init__(name=name)
        self.k = k

    def get_config(self):
        return {'k': self.k}

    def call(self, input):
        return tf.multiply(input, 2)

class QuantizedEmbedding(tf.keras.layers.Layer):
    def __init__(self, input_size, output_size, nbits=8, input_length=None, name=None):
        super(QuantizedEmbedding, self).__init__(name=name)
        self.input_size = input_size
        self.output_size = output_size
        self.input_length = input_length
        self.nbits = nbits
        vocab_size = input_size

        maxInteger = 2**nbits

        self.num_cells = int(math.ceil(vocab_size / float(maxInteger)))

    def build(self, input_shape):
        self.embeddings = self.add_weight(shape=(self.input_size, self.output_size), initializer='uniform', name='embeddings',trainable=True)

    def compute_output_shape(self, input_shape):
        if self.input_length is None:
            return input_shape + (self.output_size,)
        else:
            if isinstance(self.input_length, (list, tuple)):
                in_lens = list(self.input_length)
            else:
                in_lens = [self.input_length]
        return (input_shape[0],) + tuple(in_lens) + (self.output_size,)

    def get_config(self):
        return {
            'input_size': self.input_size,
            'output_size':self.output_size,
            'input_length':self.input_length,
            'name':self.name}


    def call(self, inputs):
        inputs = tf.cast(inputs, 'int32')
        output = tf.nn.embedding_lookup(self.embeddings, inputs)
        output = tf.cast(output, self._dtype_policy.compute_dtype)
        output = output
        return output

class LinearLayer(tf.keras.layers.Layer):
    def __init__(self, name=None, trainable=True, **kwargs):
        super(LinearLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def get_config(self):
        config = {"name":self.name}
        base_config = super(LinearLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, x):
        return x

class ScaledConvolutionalDotProduct(tf.keras.layers.Layer):
    def __init__(self, inp_size, d_model, name=None):
        super(ScaledConvolutionalDotProduct, self).__init__(name=name)
        self.d_model = d_model
        self.inp_size = inp_size
        self.softmax = activations.get('softmax')
        self.rank = 1
        self.look_back = 1
        self.strides = conv_utils.normalize_tuple(1, self.rank, 'strides')

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        kernel_shape = (self.look_back,self.inp_size,self.d_model)
        self.kernel_q = self.add_weight(
            name='kernel_q',
            shape=kernel_shape,
            initializer='glorot_uniform',
            trainable=True,
            dtype=self.dtype)

        self.kernel_k = self.add_weight(
            name='kernel_k',
            shape=kernel_shape,
            initializer='glorot_uniform',
            trainable=True,
            dtype=self.dtype)

        self.kernel_v = self.add_weight(
            name='kernel_v',
            shape=kernel_shape,
            initializer='glorot_uniform',
            trainable=True,
            dtype=self.dtype)

        self.bias_q = self.add_weight(
            name='bias_q',
            shape=(self.d_model,),
            initializer='random_normal',
            trainable=True,
            dtype=self.dtype)

        self.bias_k = self.add_weight(
            name='bias_k',
            shape=(self.d_model,),
            initializer='random_normal',
            trainable=True,
            dtype=self.dtype)

        self.bias_v = self.add_weight(
            name='bias_v',
            shape=(self.d_model,),
            initializer='random_normal',
            trainable=True,
            dtype=self.dtype)
    
        tf_strides = list(self.strides)
        self._convolution_op = functools.partial(
            nn_ops.convolution_v2,
            strides=tf_strides,
            name='conv1d')

        self._convolution_op_matmul = functools.partial(
            nn_ops.convolution_v2,
            strides=conv_utils.normalize_tuple(1, 1, 'strides'),
            name='conv1d')

    def get_config(self):
        return {'d_model': self.d_model, 'name':self.name, 'inp_size':self.inp_size}

    def call(self, q, k, v, mask):
        q = tf.expand_dims(q, axis=0)
        k = tf.expand_dims(k, axis=0)
        v = tf.expand_dims(v, axis=0)

        q = self._convolution_op(q, self.kernel_q) + self.bias_q
        k = self._convolution_op(k, self.kernel_k) + self.bias_k
        v = self._convolution_op(v, self.kernel_v) + self.bias_v
        
        q = tf.squeeze(q,axis=0)
        k = tf.squeeze(k,axis=0)
        v = tf.squeeze(v,axis=0)

        matmul_qk = tf.matmul(q,k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        if not mask is None:
            scaled_attention_logits += mask
        attention_weights = self.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights,v)
        return output

class ScaledDotProduct(tf.keras.layers.Layer):
    def __init__(self, inp_size, d_model, activation='gelu', partition_config=None, name=None):
        super(ScaledDotProduct, self).__init__(name=name)
        partition_config = deserialize_partition_config(None)
        self.cfg_json = json.dumps(partition_config)
        self.partition_config=partition_config
        self.d_model = d_model
        self.activation = activation
        self.inp_size = inp_size
        self.query_activation = activations.get(activation)
        self.key_activation = activations.get(activation)
        self.value_activation = activations.get(activation)
        self.softmax = activations.get('softmax')
        use_conv = partition_config["use_conv"]
        self.use_conv = use_conv

        self.dense_q = ConfigurableDense(self.d_model, inp_size=self.inp_size, name="query", use_conv=use_conv)
        self.dense_k = ConfigurableDense(self.d_model, inp_size=self.inp_size, name="key", use_conv=use_conv)
        self.dense_v = ConfigurableDense(self.d_model, inp_size=self.inp_size, name="value", use_conv=use_conv)

    def compute_output_shape(self, input_shape):
        return (None, input_shape[1], self.inp_size)

    def build(self, input_shape):
        pass

    def get_kernels(self):
        return [self.dense_q.get_kernels(),self.dense_k.get_kernels(),self.dense_v.get_kernels()]

    def set_kernels(self, kernels):
        self.dense_q.set_kernels(kernels[0])
        self.dense_k.set_kernels(kernels[1])
        self.dense_v.set_kernels(kernels[2])

    def get_config(self):
        #inp_size, d_model, activation='gelu', partition_config=None, name=None
        return {
            'inp_size':self.inp_size,
            'd_model': self.d_model,
            'name':self.name,
            'activation':self.activation,
            #'partition_config':self.cfg_json
        }


    def call(self, q, k, v, mask=None):
        q = self.dense_q(q)
        k = self.dense_k(k)
        v = self.dense_v(v)

        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        if not mask is None:
            scaled_attention_logits += mask
        attention_weights = self.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)
        return  output

class MultiHeadedAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, d_model, activation='gelu', name=None):
        super(MultiHeadedAttention, self).__init__(name=name)
        self.attention_heads = []
        for i in range(num_heads):
            sdp = ScaledDotProduct(d_model, int(d_model/num_heads), activation=activation, name=self.name + 'scaled_dot_product' + str(i))
            self.attention_heads.append(sdp)
        self.num_heads = num_heads
        self.activation = activation
        self.act_out = activations.get(activation)
        if activation == 'gelu':
            self.act_out = approx_gelu
        self.d_model = d_model

    def build(self, input_shape):
        for head in self.attention_heads:
            head.build(input_shape)
        self.kernel = self.add_weight(self.name + "_kernel",shape=[int(self.d_model/self.num_heads),self.d_model],
                initializer='random_normal',
                trainable=True)
        self.bias = self.add_weight(self.name + "_bias",shape=[self.d_model],
                initializer='random_normal',
                trainable=True)

    def get_config(self):
        return {'d_model': self.d_model, 'num_heads': self.num_heads, 'name':self.name, 'activation':self.activation}

    def call(self, q, k, v, mask, training=True):
        attention_outputs = []
        for head in self.attention_heads:
            attention_outputs.append(head(q, k, v, mask))
        x = attention_outputs[0]
        if self.num_heads>1:
            x = tf.keras.layers.concatenate(attention_outputs)
        x = self.act_out(tf.matmul(x, self.kernel) + self.bias)
        return x


class BERTMultiHeadedAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, d_model, rate=0.1, partition_config=None, activation='gelu', name=None):
        super(BERTMultiHeadedAttention, self).__init__(name=name)
        if partition_config==None:
            partition_config=DEFAULT_PARTITION_CONFIG
        self.partition_config = partition_config
        self.attention_heads = []
        self.use_conv = partition_config["use_conv"]
        for i in range(num_heads):
            sdp = ScaledDotProduct(d_model, int(d_model/num_heads), partition_config=partition_config, name=self.name + 'sdp_' + str(i))
            self.attention_heads.append(sdp)
        self.num_heads = num_heads
        self.rate = rate
        self.activation = activation
        self.act_out = activations.get(activation)
        self.d_model = d_model
        self.dropout = tf.keras.layers.Dropout(rate)
        self.mha_ffn = ConfigurableDense(self.d_model, inp_size=self.d_model, use_conv=self.use_conv, name=self.name + "attention_output")

    def build(self, input_shape):
        pass

    def get_config(self):
        return {
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'name':self.name,
            'activation':self.activation,
            'rate':self.rate,
            'partition_config':self.partition_config
            }

    def get_kernels(self):
        ret = [self.mha_ffn.get_kernels()]
        for head in self.attention_heads:
            ret+=head.get_kernels()
        return ret

    def set_kernels(self, kernels):
        self.mha_ffn.set_kernels(kernels[0])
        for i,head in enumerate(self.attention_heads):
            #TODO use a dictionary instead to make this more easily expandable
            head.set_kernels(kernels[1+i*len(head.get_kernels()):1+(i+1)*len(head.get_kernels())])

    def call(self, q, k, v, mask, training=True):
        attention_outputs = []
        for head in self.attention_heads:
            attention_outputs.append(head(q, k, v, mask))
        x = attention_outputs[0]
        if self.num_heads>1:
            x = tf.keras.layers.concatenate(attention_outputs)
        x = self.mha_ffn(x)
        x = self.dropout(x, training=training)
        return x

class ConfigurableDense(tf.keras.layers.Layer):
    def __init__(self, size, use_conv=False, inp_size=None, use_bias=True, activation=None, name=None):
        super(ConfigurableDense, self).__init__(name=name)
        self.size = size
        self.use_conv = use_conv
        self.use_bias = use_bias
        self.inp_size = inp_size
        self.activation = activation
        self.act_out = activations.get(activation)
        if activation=="gelu":
            self.act_out=ApproxGelu()

    def get_config(self):
        return {
            'name': self.name,
            'size': self.size,
            'use_conv': self.use_conv,
            'use_bias':self.use_bias,
            'inp_size': self.inp_size,
            'activation':self.activation,
            }

    def build(self, input_shape):
        inp_size = self.inp_size
        if inp_size is None:
            inp_size = input_shape[-1]
        if not self.use_conv:
            self.kernel = self.add_weight("kernel",shape=[self.inp_size,self.size],
                    initializer='random_normal',
                    trainable=True)
        else:
            kernel_shape = (1,inp_size,self.size)
            self.kernel = self.add_weight(
            name='kernel',
            shape=kernel_shape,
            initializer='glorot_uniform',
            trainable=True)
        
        if self.use_bias:
            self.bias = self.add_weight("bias",shape=[self.size],
                initializer='random_normal',
                trainable=True)

        self._convolution_op = functools.partial(
            nn_ops.convolution_v2,
            strides=list(conv_utils.normalize_tuple(1, 1, 'strides')),
            name='conv1d')

    def get_kernels(self):
        return self.kernel

    def set_kernels(self, kernel):
        self.kernel = kernel

    def get_activations(self):
        if activation=="gelu":
            return self.act_out.get_activations()
        return self.act_out

    def set_activations(self, activation):
        if activation=="gelu":
            self.act_out.set_activations(activation)
            return
        self.act_out = activation

    
    def call(self, x):
        if not self.use_conv:
            out = tf.matmul(x, self.kernel)
        else:
            out = tf.expand_dims(x, axis=0)
            out = self._convolution_op(out, self.kernel)
            out = tf.squeeze(out,axis=0)
        if self.use_bias:
            out += self.bias
        if self.activation is not None:
            out = self.act_out(out)
        return out


class PartitionLayer(tf.keras.layers.Layer):
    def __init__(self, output_size, input_size, num_layers=1, partition_output=True, rank=2, use_conv=False, use_bias=True, activation=None, name=None):
        super(PartitionLayer, self).__init__(name=name)
        if partition_output:
          assert output_size%num_layers==0
        else:
          assert input_size%num_layers==0
        self.num_layers = num_layers
        self.output_size = output_size
        self.rank = rank
        self.use_conv = use_conv
        self.input_size=input_size
        self.activation = activation
        self.use_bias = use_bias
        self.partition_output = partition_output
        self.fcs = []
        for i in range(num_layers):
          if partition_output:
            self.fcs.append(ConfigurableDense(int(output_size/num_layers), inp_size=input_size, activation=activation, use_conv=use_conv, name="partition_out" + str(i)))
          else:
            self.fcs.append(ConfigurableDense(output_size, inp_size=int(input_size/num_layers), activation=activation, use_conv=use_conv, use_bias=False, name=str(i)))

    def build(self, input_shape):
      if self.use_bias and not self.partition_output:
        self.bias = self.add_weight("bias",shape=[self.output_size],
            initializer='random_normal',
            trainable=True)

    def get_config(self):
        return {
            'name':self.name,
            'use_conv':self.use_conv
            }


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
        if self.use_bias:
            output+=self.bias
        return output
      elif rank==3:
        output = self.fcs[0](x[:,:,0:partition_size])
        for i in range(1,self.num_layers):
          output+=self.fcs[i](x[:,:,partition_size*(i):partition_size*(i+1)])
        if self.use_bias:
          output+=self.bias
        return output
      return x


class PartitionEmbedding(tf.keras.layers.Layer):
  def __init__(self, vocab_Size, emb_size, n_partitions=1, name=None):
      super(PartitionEmbedding, self).__init__(name=name)
      self.vocab_size = vocab_Size
      self.emb_size = emb_size
      self.n_partitions = n_partitions
      self.partition_size = int(self.emb_size/n_partitions)
      assert self.emb_size % n_partitions == 0
      self.embeddings = [tf.keras.layers.Embedding(vocab_Size, self.partition_size, name="partition" + str(i)) for i in range(n_partitions)]

  def call(self, x):
      outputs = []
      for embedding in self.embeddings:
          outputs.append(embedding(x))
      x = outputs[0]
      if self.n_partitions>1:
          x = tf.keras.layers.concatenate(outputs)
      return x


class BertEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, seq_len, n_segments, d_model, n_partitions=1, name=None):
        super(BertEmbedding, self).__init__(name=name)

        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.n_segments = n_segments
        self.d_model = d_model
        self.n_partitions = n_partitions


        #self.word_embeddings = tf.keras.layers.Embedding(vocab_size, d_model, name="word_embeddings")
        #self.position_embedding = tf.keras.layers.Embedding(seq_len, d_model, name="position_embeddings")
        #self.type_embeddings = tf.keras.layers.Embedding(n_segments, d_model, name="type_embeddings")
        
        self.word_embeddings = PartitionEmbedding(vocab_size, d_model, n_partitions=n_partitions, name="word_embeddings")
        self.position_embedding = PartitionEmbedding(seq_len, d_model, n_partitions=n_partitions, name="position_embeddings")
        self.type_embeddings = PartitionEmbedding(n_segments, d_model, n_partitions=n_partitions, name="type_embeddings")
        self.norm = DynamicLayerNormalization(epsilon=1e-12, name="layer_normalization")

    def get_config(self):
        return {
            'vocab_size': self.vocab_size,
            'seq_len': self.seq_len,
            'n_segments':self.n_segments,
            'd_model':self.d_model,
            'n_partitions':self.n_partitions,
            'name':self.name
            }

    def build(self, input_shape):
        pass

    def call(self, x, seg):
        seq_len = tf.shape(x)[1]
        batch_size = tf.shape(x)[0]
        pos = tf.range(seq_len)
        pos = tf.expand_dims(pos, 0)
        pos = tf.broadcast_to(pos, (batch_size, seq_len))
        embedding = self.word_embeddings(x) + self.type_embeddings(seg) + self.position_embedding(pos)
        return self.norm(embedding)

class BERT(tf.keras.layers.Layer):
    def __init__(self, n_layers, num_heads, vocab_size, seq_len, n_segments, d_model, intermediate_size, rate=0.1, partition_config=None, activation='gelu', name=None):
        super(BERT, self).__init__(name=name)
        if partition_config==None:
            partition_config=DEFAULT_PARTITION_CONFIG
        self.partition_config=partition_config
        self.rate = rate
        self.n_layers = n_layers
        self.num_heads = num_heads
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.n_segments = n_segments
        self.use_conv = partition_config["use_conv"]
        self.d_model = d_model
        self.intermediate_size = intermediate_size
        self.n_partitions = partition_config["intermediate_partitions"]

        embedding_partitions = partition_config["embedding_partitions"]


        self.embedding = BertEmbedding(vocab_size, seq_len, n_segments, d_model, n_partitions=embedding_partitions)
        self.enc_layers = [BertEncoder(num_heads, d_model, intermediate_size, rate=rate, activation=activation, partition_config=partition_config, name="layer_" + str(i)) 
                        for i in range(n_layers)]

        self.activation = activation
        self.act_out = activations.get(activation)
        if activation == 'gelu':
            self.act_out = ApproxGelu()

        self.pooler_ffn = ConfigurableDense(self.d_model, inp_size=self.d_model, use_conv=self.use_conv, name = self.name + "pooler_transform")

    def get_config(self):
        return {
            'n_layers':self.n_layers,
            'num_heads': self.num_heads,
            'vocab_size': self.vocab_size,
            'seq_len': self.seq_len,
            'n_segments':self.n_segments,
            'name':self.name,
            'd_model':self.d_model,
            'intermediate_size':self.intermediate_size,
            'activation':self.activation,
            'rate':self.rate,
            'partition_config':self.partition_config
            }
        
    def get_kernels(self):
        kernels = [self.pooler_ffn.get_kernels()]
        for encoder in self.enc_layers:
            kernels += encoder.get_kernels()
        return kernels

    def set_kernels(self, kernels):
        #self.pooler_ffn.set_kernels(kernels[0])
        for i, encoder in enumerate(self.enc_layers):
            enc_len = len(encoder.get_kernels())
            encoder.set_kernels(kernels[1+(i*enc_len):1+((i+1)*enc_len)])
            

    def build(self, input_shape):
        pass

    def call(self, x, seg, mask, training=True):
        mask = tf.expand_dims(mask, axis=1)
        mask = mask*1e-9

        x = self.embedding(x,seg)
        for layer in self.enc_layers:
            x = layer(x, mask)
        x = x[:,0]
        x = self.act_out(self.pooler_ffn(x))
        return x
        #return {"pooled_output":x}

class BertEncoder(tf.keras.layers.Layer):
    def __init__(self, num_heads, d_model, intermediate_size, rate=0.1, activation='gelu', partition_config=None, name=None):
        super(BertEncoder, self).__init__(name=name)
        if partition_config==None:
            partition_config=DEFAULT_PARTITION_CONFIG

        self.partition_config=partition_config
        self.d_model = d_model
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.rate = rate
        self.use_conv = partition_config["use_conv"]

        self.mha = BERTMultiHeadedAttention(num_heads, d_model, activation=activation, partition_config=partition_config, name = "mha")

        self.layernorm1 = DynamicLayerNormalization(epsilon=1e-12, name="attention_layer_norm")
        self.layernorm2 = DynamicLayerNormalization(epsilon=1e-12, name="output_layer_norm")

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.activation = activation

        #self.dff = PartitionLayer(self.intermediate_size, self.d_model, num_layers=partition_config["intermediate_partitions"], activation=activation, use_conv=partition_config["use_conv"], name="intermediate")#ConfigurableDense(self.intermediate_size, inp_size=self.d_model, use_conv=use_conv, name=self.name + "/intermediate/")
        #self.out_ffn = PartitionLayer(self.d_model, self.intermediate_size, partition_output=False, num_layers=partition_config["fc_out_partitions"], use_conv=partition_config["use_conv"], rank=3, name="out")
        self.dff = ConfigurableDense(self.intermediate_size, inp_size=self.d_model, use_conv=partition_config["use_conv"], activation=activation, name="intermediate")
        self.out_ffn = ConfigurableDense(self.d_model, inp_size=self.intermediate_size, use_conv=partition_config["use_conv"], name="out")


        self.intermediate_partitions = 1
        self.use_partitions=True

        self.activation1 = activations.get(activation)
        if activation == 'gelu':
            self.activation1 = approx_gelu


    def get_config(self):
        return {
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'intermediate_size':self.intermediate_size,
            'rate':self.rate,
            'partition_config':self.partition_config,
            'name':self.name,
            'activation':self.activation,
            }

            #num_heads, d_model, intermediate_size, rate=0.1, activation='gelu', partition_config=None, name=None

    def get_kernels(self):
        return [self.dff.get_kernels(), self.out_ffn.get_kernels()] + self.mha.get_kernels()

    def set_kernels(self, kernels):
        #self.dff.set_kernels(kernels[0])
        #self.out_ffn.set_kernels(kernels[1])
        #self.mha.set_kernels(kernels[2:])
        pass

    def get_activations(self):
        return self.dff.get_activations()

    def build(self, input_shape):
        pass

    def call(self, x, mask, training=True):
        attn_output = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        #out1 = attn_output
        ffn_output1 = self.dff(out1)
        ffn_output2 = self.out_ffn(ffn_output1)
        ffn_output3 = self.dropout2(ffn_output2, training=training)
        #out2 = ffn_output3
        out2 = self.layernorm2(out1 + ffn_output3)
        return out2

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, num_heads, d_model, dff, partition_config=None, rate=0.1, name=None):
        super(EncoderLayer, self).__init__(name=name)
        if partition_config==None:
            partition_config = DEFAULT_PARTITION_CONFIG

        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate

        self.mha = BERTMultiHeadedAttention(num_heads, d_model, activation='relu', partition_config=partition_config, name = "mha")
        
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-12)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-12)
        
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

        self.intr = ConfigurableDense(self.dff, inp_size=self.d_model, activation='relu', use_conv=partition_config["use_conv"])
        self.fc_out = ConfigurableDense(self.d_model, inp_size=self.dff, activation='relu', use_conv=partition_config["use_conv"])

    def get_config(self):
        return {
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dff':self.dff,
            'rate':self.rate,
            'name':self.name
            }

    def build(self, input_shape):
        pass

    def call(self, x, mask, training=True):
        attn_output = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        
        ffn_output1 = self.intr(out1)
        ffn_output2 = self.fc_out(ffn_output1)
        ffn_output3 = self.dropout2(ffn_output2, training=training)
        out2 = self.layernorm2(out1 + ffn_output3)
        
        return out2

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, num_heads, d_model, dff, partition_config=None, rate=0.1, name=None):
        super(DecoderLayer, self).__init__(name=name)
        if partition_config==None:
            partition_config = DEFAULT_PARTITION_CONFIG
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate

        self.mha1 = BERTMultiHeadedAttention(num_heads, d_model, activation='relu', partition_config=partition_config, name = "mha")
        self.mha2 = BERTMultiHeadedAttention(num_heads, d_model, activation='relu', partition_config=partition_config, name = "mha")

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-12)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-12)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-12)
        
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

        self.intr = ConfigurableDense(self.dff, inp_size=self.d_model, activation='relu', use_conv=partition_config["use_conv"])
        self.fc_out = ConfigurableDense(self.d_model, inp_size=self.dff, activation='relu', use_conv=partition_config["use_conv"])

    def get_config(self):
        return {
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dff':self.dff,
            'rate':self.rate,
            'name':self.name
            }

    def build(self, input_shape):
        pass

    def call(self, x, enc_output, combined_mask, pad_mask, training=True):
        attn1 = self.mha1(x, x, x, combined_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)
        
        attn2 = self.mha2(out1, enc_output, enc_output, pad_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)
        
        #ffn_output = relu(tf.matmul(out2, self.kernel_dff) + self.bias_dff)
        #ffn_output = relu(tf.matmul(ffn_output, self.kernel_out) + self.bias_out)
        ffn_output = self.intr(out2)
        ffn_output = self.fc_out(ffn_output)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)
        
        return out3

class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, num_heads, d_model, dff, input_vocab_size,
            maximum_position_encoding, partition_config=None, rate=0.1, name=None):
        super(Encoder, self).__init__(name=name)
        if partition_config==None:
            partition_config = DEFAULT_PARTITION_CONFIG
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dff = dff
        self.input_vocab_size = input_vocab_size
        self.maximum_position_encoding = maximum_position_encoding
        self.rate = rate
        
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, 
                                                self.d_model)

        self.enc_layers = [EncoderLayer(num_heads, d_model, dff, rate=rate, partition_config=partition_config, name=self.name + str(i)) 
                        for i in range(num_layers)]
    
        self.dropout = tf.keras.layers.Dropout(rate)

    def get_config(self):
        return {
            'num_layers': self.num_layers,
            'input_vocab_size': self.input_vocab_size,
            'maximum_position_encoding': self.maximum_position_encoding,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dff':self.dff,
            'rate':self.rate,
            'name':self.name
            }

    def build(self, input_shape):
        for i in range(self.num_layers):
            self.enc_layers[i].build([self.d_model])


    def call(self, x, mask, training=True):

        # adding embedding and position encoding.
        seq_len = tf.shape(x)[1]
        #mask = tf.where(tf.equal(x,0), tf.ones_like(x)*-1e9, tf.zeros_like(x))
        #mask = tf.cast(tf.math.equal(x, 0), tf.float32)
        #mask = mask[:, tf.newaxis, :]
        #mask = tf.expand_dims(mask, axis=1)

        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)
        
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, mask, training=training)
        return x


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, num_heads, d_model, dff, target_vocab_size,
            maximum_position_encoding, partition_config=None, rate=0.1, name=None):
        super(Decoder, self).__init__(name=name)
        if partition_config==None:
            partition_config = DEFAULT_PARTITION_CONFIG
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dff = dff
        self.target_vocab_size = target_vocab_size
        self.maximum_position_encoding = maximum_position_encoding
        self.rate = rate
        
        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, 
                                                self.d_model)

        self.dec_layers = [DecoderLayer(num_heads, d_model, dff, rate=rate, name=self.name + str(i), partition_config=partition_config) 
                        for i in range(num_layers)]
    
        self.dropout = tf.keras.layers.Dropout(rate)

    def get_config(self):
        return {
            'num_layers': self.num_layers,
            'target_vocab_size': self.target_vocab_size,
            'maximum_position_encoding': self.maximum_position_encoding,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dff':self.dff,
            'rate':self.rate,
            'name':self.name
            }

    def build(self, input_shape):
        for i in range(self.num_layers):
            self.dec_layers[i].build([self.d_model])


    def call(self, x, enc_output, enc_mask, combined_mask, training=True):

        seq_len = tf.shape(x)[1]

        #look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        #dec_target_padding_mask = tf.where(tf.equal(x,0), tf.ones_like(x), tf.zeros_like(x))
        #dec_target_padding_mask = tf.expand_dims(dec_target_padding_mask, axis=1)
        #combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)*-1e9

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)
        
        for i in range(self.num_layers):
            x = self.dec_layers[i](x, enc_output, combined_mask, enc_mask, training=training)
        return x