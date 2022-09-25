#
# Author: Brendan Reidy
# Email: bcreidy@email.sc.edu
# Date created: May 5 2020
# Last Modified: Sep 11 2022
#

from tokenize import Funny
import tensorflow as tf
import sys
import numpy as np
import utils.MaskUtils
import tensorflow_model_optimization as tfmot
from keras import activations
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.ops import nn_ops
import functools
import math

LastValueQuantizer = tfmot.quantization.keras.quantizers.LastValueQuantizer
MovingAverageQuantizer = tfmot.quantization.keras.quantizers.MovingAverageQuantizer

def relu(x):
    y = tf.constant([0.])
    return tf.math.maximum(x, y)

erf_a = -0.2888
erf_b = -1.769
erf_c = 1

#abs function is not supported on edge tpu
def tf_abs(x):
    #return tf.math.abs(x)
    return tf_sign(x)*x

#sign function is not supported in tflite
def tf_sign(x):
    #return tf.math.sign(x)
    return tf.tanh(x*1e3)

def approx_erf(x):
    return tf_sign(x)*(erf_a * (tf.math.minimum(tf_abs(x), -erf_b)+erf_b)**2 + erf_c)

def approx_gelu(x):
    return x*0.5*(1+approx_erf(x/math.sqrt(2)))


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
    def __init__(self, name=None):
        super(LinearLayer, self).__init__(name=name)

    def build(self, input_shape):
        pass

    def get_config(self):
        return {
            'name':self.name}

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
    def __init__(self, inp_size, d_model, activation='gelu', use_conv=False, name=None):
        super(ScaledDotProduct, self).__init__(name=name)
        self.d_model = d_model
        self.activation = activation
        self.inp_size = inp_size
        self.query_activation = activations.get(activation)
        self.key_activation = activations.get(activation)
        self.value_activation = activations.get(activation)
        self.softmax = activations.get('softmax')
        self.use_conv = use_conv

        self.dense_q = Dense(self.d_model, inp_size=self.inp_size, name="query", use_conv=use_conv)
        self.dense_k = Dense(self.d_model, inp_size=self.inp_size, name="key", use_conv=use_conv)
        self.dense_v = Dense(self.d_model, inp_size=self.inp_size, name="value", use_conv=use_conv)

    def build(self, input_shape):
        """
        self.kernel_q = self.add_weight(self.name + "_kernel_q",shape=[self.inp_size,self.d_model],
                initializer='random_normal',
                trainable=True)
        self.bias_q = self.add_weight(self.name + "_bias_q",shape=[self.d_model],
                initializer='random_normal',
                trainable=True)

        self.kernel_k = self.add_weight(self.name + "_kernel_k",shape=[self.inp_size,self.d_model],
                initializer='random_normal',
                trainable=True)
        self.bias_k = self.add_weight(self.name + "_bias_k",shape=[self.d_model],
                initializer='random_normal',
                trainable=True)

        self.kernel_v = self.add_weight(self.name + "_kernel_v",shape=[self.inp_size,self.d_model],
                initializer='random_normal',
                trainable=True)
        self.bias_v = self.add_weight(self.name + "_bias_v",shape=[self.d_model],
                initializer='random_normal',
                trainable=True)
        """

    def get_config(self):
        return {'d_model': self.d_model, 'name':self.name, 'activation':self.activation, 'inp_size':self.inp_size, 'use_conv':self.use_conv}


    def call(self, q, k, v, mask=None):
        #q = tf.matmul(q, self.kernel_q) + self.bias_q
        #k = tf.matmul(k, self.kernel_k) + self.bias_k
        #v = tf.matmul(v, self.kernel_v) + self.bias_v
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

class TPUAcceleratedMultiHeadedAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, d_model, name=None):
        super(TPUAcceleratedMultiHeadedAttention, self).__init__(name=name)
        self.attention_heads = []
        for i in range(num_heads):
            sdp = ScaledConvolutionalDotProduct(d_model, name=self.name + 'scaled_dot_product' + str(i))
            self.attention_heads.append(sdp)
        self.num_heads = num_heads
        self.relu = activations.get('relu')
        self.d_model = d_model

    def build(self, input_shape):
        for head in self.attention_heads:
            head.build(input_shape)
        self.kernel = self.add_weight(self.name + "_kernel",shape=[self.d_model*self.num_heads,self.d_model],
                initializer='random_normal',
                trainable=True)
        self.bias = self.add_weight(self.name + "_bias",shape=[self.d_model],
                initializer='random_normal',
                trainable=True)

    def get_config(self):
        return {'d_model': self.d_model, 'num_heads': self.num_heads, 'name':self.name}

    def call(self, q, k, v, mask):
        batch_size = tf.shape(q)[0]

        attention_outputs = []
        for head in self.attention_heads:
            attention_outputs.append(head(q, k, v, mask))
        x = attention_outputs[0]
        if self.num_heads>1:
            x = tf.keras.layers.concatenate(attention_outputs)
        x = self.relu(x)
        x = self.relu(tf.matmul(x, self.kernel) + self.bias)
        return x

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
    def __init__(self, num_heads, d_model, rate=0.1, activation='gelu', use_conv=False, name=None):
        super(BERTMultiHeadedAttention, self).__init__(name=name)
        self.attention_heads = []
        self.use_conv = use_conv
        for i in range(num_heads):
            sdp = ScaledDotProduct(d_model, int(d_model/num_heads), use_conv=use_conv, name=self.name + 'sdp_' + str(i))
            self.attention_heads.append(sdp)
        self.num_heads = num_heads
        self.rate = rate
        self.activation = activation
        self.act_out = activations.get(activation)
        self.d_model = d_model
        self.dropout = tf.keras.layers.Dropout(rate)
        self.mha_ffn = Dense(self.d_model, inp_size=self.d_model, use_conv=self.use_conv, name=self.name + "attention_output")

    def build(self, input_shape):
        #for head in self.attention_heads:
            #head.build(input_shape)
        """
        self.kernel = self.add_weight(self.name + "attention_output_kernel",shape=[input_shape[-1],self.d_model],
                initializer='random_normal',
                trainable=True)
        self.bias = self.add_weight(self.name + "attention_output_bias",shape=[self.d_model],
                initializer='random_normal',
                trainable=True)
        """

    def get_config(self):
        return {'d_model': self.d_model, 'num_heads': self.num_heads, 'use_conv':self.use_conv, 'name':self.name, 'activation':self.activation, 'rate':self.rate}

    def call(self, q, k, v, mask, training=True):
        attention_outputs = []
        for head in self.attention_heads:
            attention_outputs.append(head(q, k, v, mask))
        x = attention_outputs[0]
        if self.num_heads>1:
            x = tf.keras.layers.concatenate(attention_outputs)
        #x = tf.matmul(x, self.kernel) + self.bias
        x = self.mha_ffn(x)
        x = self.dropout(x, training=training)
        return x

class Dense(tf.keras.layers.Layer):
    def __init__(self, size, use_conv=False, inp_size=None, use_bias=True, activation=None, name=None):
        super(Dense, self).__init__(name=name)
        self.size = size
        self.use_conv = use_conv
        self.use_bias = use_bias
        self.inp_size = inp_size
        self.activation = activation
        self.act_out = activations.get(activation)

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




class BertEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, seq_len, n_segments, d_model, name=None):
        super(BertEmbedding, self).__init__(name=name)

        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.n_segments = n_segments
        self.d_model = d_model

        self.word_embeddings = tf.keras.layers.Embedding(vocab_size, d_model, name="word_embeddings")
        self.position_embedding = tf.keras.layers.Embedding(seq_len, d_model, name="position_embeddings")
        self.type_embeddings = tf.keras.layers.Embedding(n_segments, d_model, name="type_embeddings")
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-12, name="layer_normalization")

    def get_config(self):
        return {
            'vocab_size': self.vocab_size,
            'seq_len': self.seq_len,
            'n_segments':self.n_segments,
            'd_model':self.d_model,
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
    def __init__(self, n_layers, num_heads, vocab_size, seq_len, n_segments, d_model, intermediate_size, rate=0.1, intermediate_partitions=1, use_conv=False, activation='gelu', name=None):
        super(BERT, self).__init__(name=name)
        self.n_layers = n_layers
        self.num_heads = num_heads
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.n_segments = n_segments
        self.use_conv = use_conv
        self.d_model = d_model
        self.intermediate_partitions = intermediate_partitions
        self.intermediate_size = intermediate_size


        self.embedding = BertEmbedding(vocab_size, seq_len, n_segments, d_model)
        self.enc_layers = [BertEncoder(num_heads, d_model, intermediate_size, rate=rate, intermediate_partitions=intermediate_partitions, activation=activation, use_conv=use_conv, name="layer_" + str(i)) 
                        for i in range(n_layers)]

        self.activation = activation
        self.act_out = activations.get(activation)
        if activation == 'gelu':
            self.act_out = approx_gelu

        self.pooler_ffn = Dense(self.d_model, inp_size=self.d_model, use_conv=use_conv, name = self.name + "pooler_transform")

    def get_config(self):
        return {
            'n_layers':self.n_layers,
            'num_heads': self.num_heads,
            'vocab_size': self.vocab_size,
            'seq_len': self.seq_len,
            'n_segments':self.n_segments,
            'use_conv':self.use_conv,
            'name':self.name,
            'd_model':self.d_model,
            'intermediate_size':self.intermediate_size,
            'activation':self.activation,
            'intermediate_partitions':self.intermediate_partitions
            }
    def build(self, input_shape):
        pass

    def set_partitions(self,n_partitions):
        assert self.intermediate_partitions==1

        self.intermediate_partitions = n_partitions
        self.use_partitions = True
        for enc in self.enc_layers:
            enc.set_partitions(n_partitions)

    def call(self, x, seg, mask, training=True):
        mask = tf.expand_dims(mask, axis=1)
        mask = mask*1e-9

        x = self.embedding(x,seg)
        for layer in self.enc_layers:
            x = layer(x, mask)
        x = x[:,0]
        #x = self.act_out(tf.matmul(x, self.pooler_transform) + self.pooler_transform_bias)
        x = self.act_out(self.pooler_ffn(x))
        return x
        #return {"pooled_output":x}

class BertEncoder(tf.keras.layers.Layer):
    def __init__(self, num_heads, d_model, intermediate_size, rate=0.1, intermediate_partitions=1, activation='gelu', use_conv=False, name=None):
        super(BertEncoder, self).__init__(name=name)

        self.d_model = d_model
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.rate = rate
        self.use_conv = use_conv
        self.intermediate_partitions = intermediate_partitions

        self.mha = BERTMultiHeadedAttention(num_heads, d_model, activation=activation, use_conv=use_conv, name = "mha")

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-12, name="attention_layer_norm")
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-12, name="output_layer_norm")

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.activation = activation
        self.use_partitions = (not self.intermediate_partitions==1)

        self.dff = Dense(self.intermediate_size, inp_size=self.d_model, use_conv=use_conv, name=self.name + "/intermediate/")
        self.out_ffn = Dense(self.d_model, inp_size=self.intermediate_size, use_conv=use_conv, name=self.name + "/out/")

        self.activation1 = activations.get(activation)
        if activation == 'gelu':
            self.activation1 = approx_gelu


    def get_config(self):
        return {
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'intermediate_size':self.intermediate_size,
            'rate':self.rate,
            'use_conv':self.use_conv,
            'name':self.name,
            'activation':self.activation,
            'intermediate_partitions':self.intermediate_partitions
            }

    def build_partitions(self):
        assert self.intermediate_size % self.intermediate_partitions == 0

        partition_size = int(self.intermediate_size / self.intermediate_partitions)

        curr_dff = self.kernel_dff
        curr_bias_dff = self.bias_dff
        curr_out = self.kernel_out

        self.kernel_dff_part = []
        self.bias_dff_part = []
        self.kernel_out_part = []
        for i in range(self.intermediate_partitions):
            self.kernel_dff_part.append(self.add_weight(self.name + "/intermediate_partition_" + str(i) + "/kernel",shape=[self.d_model,partition_size],
                    initializer='random_normal',
                    trainable=True))
            self.bias_dff_part.append(self.add_weight(self.name + "/intermediate_partition_" + str(i) + "/bias",shape=[partition_size],
                    initializer='random_normal',
                    trainable=True))
            self.kernel_out_part.append(self.add_weight(self.name + "/kernel_out_partition_" + str(i) + "",shape=[partition_size,self.d_model],
                    initializer='random_normal',
                    trainable=True))

        for i in range(self.intermediate_partitions):
            temp_kernel_dff = curr_dff[:,i*partition_size:(i+1)*partition_size]
            self.kernel_dff_part[i].assign(temp_kernel_dff)

            temp_bias_dff = curr_bias_dff[i*partition_size:(i+1)*partition_size]
            self.bias_dff_part[i].assign(temp_bias_dff)

            temp_kernel_out = curr_out[i*partition_size:(i+1)*partition_size,:]
            self.kernel_out_part[i].assign(temp_kernel_out)

    def build_standard(self):
        self.kernel_out = self.add_weight(self.name + "/kernel_out",shape=[self.intermediate_size,self.d_model],
                initializer='random_normal',
                trainable=True)



    def build(self, input_shape):
        pass

    def compute_ffn_partition(self,x):
        ffn_outputs = []
        for i in range(self.intermediate_partitions):
            ffn_output1 = self.activation1(tf.matmul(x, self.kernel_dff_part[i]) + self.bias_dff_part[i])
            ffn_outputs.append(ffn_output1)
        ffn_output2 = tf.matmul(ffn_outputs[0], self.kernel_out_part[0])
        for i in range(1,self.intermediate_partitions):
            ffn_output2 += tf.matmul(ffn_outputs[i], self.kernel_out_part[i])
        ffn_output2 = ffn_output2 + self.bias_out
        return ffn_output2

    def compute_ffn_standard(self,x):
        ffn_output1 = self.activation1(tf.matmul(x, self.kernel_dff) + self.bias_dff)
        ffn_output2 = tf.matmul(ffn_output1, self.kernel_out) + self.bias_out
        return ffn_output2

    def compute_ffn(self,x):
        if not self.use_partitions:
            return self.compute_ffn_standard(x)
        return self.compute_ffn_partition(x)

    def set_partitions(self, n_partitions):
        assert self.intermediate_partitions==1

        self.intermediate_partitions = n_partitions
        self.use_partitions = True
        self.build_partitions()

    def call(self, x, mask, training=True):
        attn_output = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output) # (batch_Size, seq_len, d_model)
        #"""
        #ffn_output1 = self.activation1(tf.matmul(out1, self.kernel_dff) + self.bias_dff)
        ffn_output1 = self.activation1(self.dff(out1))
        ffn_output2 = self.out_ffn(ffn_output1)
        #ffn_output2 = tf.matmul(ffn_output1, self.kernel_out) + self.bias_out
        #"""
        #ffn_output2 = self.compute_ffn(out1)
        ffn_output3 = self.dropout2(ffn_output2, training=training)
        out2 = self.layernorm2(out1 + ffn_output3)
        return out2

class IntLayerNorm(tf.keras.layers.Layer):
    """
    Tensorflow implmentation of integer-only layer norm
    https://github.com/kssteven418/I-BERT
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/ibert/quant_modules.py

    """
    
    def __init__(self, epsilon=1e-6, name=None):
        super(IntLayerNorm, self).__init__(name=name)
        self.epsilon = epsilon
        self.max_bit = 32
        self.quantized = False

    def get_config(self):
        return {
            'epsilon': self.epsilon,
            'name':self.name
            }

    def build(self, input_shape):
        #gamma
        input_shape = tf.TensorShape(input_shape)
        self.weight = self.add_weight("gamma",shape=[input_shape[-1]],
                initializer='random_normal',
                trainable=True)

        #beta
        self.bias = self.add_weight("beta",shape=[input_shape[-1]],
                initializer='random_normal',
                trainable=True)

    def call(self, x):
        mean, var = tf.nn.moments(x, -1, keepdims=True)
        y = x - mean
        x = tf.math.floor(y / tf.sqrt(self.epsilon + var))
        x = x * self.weight + self.bias
        return x
    

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, num_heads, d_model, dff, rate=0.1, name=None):
        super(EncoderLayer, self).__init__(name=name)

        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate

        self.mha = MultiHeadedAttention(num_heads, dff, name = self.name + "_mha_")

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-12)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-12)
        
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def get_config(self):
        return {
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dff':self.dff,
            'rate':self.rate,
            'name':self.name
            }

    def build(self, input_shape):

        self.mha.build(input_shape)

        self.kernel_dff = self.add_weight(self.name + "_kernel_dff",shape=[self.d_model,self.dff],
                initializer='random_normal',
                trainable=True)
        self.bias_dff = self.add_weight(self.name + "_bias_dff",shape=[self.dff],
                initializer='random_normal',
                trainable=True)

        self.kernel_out = self.add_weight(self.name + "kernel_out",shape=[self.dff,self.d_model],
                initializer='random_normal',
                trainable=True)
        self.bias_out = self.add_weight(self.name + "bias_out",shape=[self.d_model],
                initializer='random_normal',
                trainable=True)

    def call(self, x, mask, training=True):
        attn_output = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        
        ffn_output1 = relu(tf.matmul(out1, self.kernel_dff) + self.bias_dff)
        ffn_output2 = relu(tf.matmul(ffn_output1, self.kernel_out) + self.bias_out)
        ffn_output3 = self.dropout2(ffn_output2, training=training)
        out2 = self.layernorm2(out1 + ffn_output3)
        
        return out2

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, num_heads, d_model, dff, rate=0.1, name=None):
        super(DecoderLayer, self).__init__(name=name)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate

        self.mha1 = MultiHeadedAttention(num_heads, d_model)
        self.mha2 = MultiHeadedAttention(num_heads, d_model)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-12)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-12)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-12)
        
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def get_config(self):
        return {
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dff':self.dff,
            'rate':self.rate,
            'name':self.name
            }

    def build(self, input_shape):

        self.mha1.build([self.d_model])
        self.mha2.build([self.d_model])

        self.kernel_dff = self.add_weight(self.name + "_kernel_dff",shape=[self.d_model,self.dff],
                initializer='random_normal',
                trainable=True)
        self.bias_dff = self.add_weight(self.name + "_bias_dff",shape=[self.dff],
                initializer='random_normal',
                trainable=True)

        self.kernel_out = self.add_weight(self.name + "kernel_out",shape=[self.dff,self.d_model],
                initializer='random_normal',
                trainable=True)
        self.bias_out = self.add_weight(self.name + "bias_out",shape=[self.d_model],
                initializer='random_normal',
                trainable=True)

    def call(self, x, enc_output, combined_mask, pad_mask, training=True):
        attn1 = self.mha1(x, x, x, combined_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)
        
        attn2 = self.mha2(out1, enc_output, enc_output, pad_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)
        
        ffn_output = relu(tf.matmul(out2, self.kernel_dff) + self.bias_dff)
        ffn_output = relu(tf.matmul(ffn_output, self.kernel_out) + self.bias_out)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)
        
        return out3

class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, num_heads, d_model, dff, input_vocab_size,
            maximum_position_encoding, rate=0.1, name=None):
        super(Encoder, self).__init__(name=name)
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

        self.enc_layers = [EncoderLayer(num_heads, d_model, dff, rate, name=self.name + str(i)) 
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


    def call(self, x, training=True):

        # adding embedding and position encoding.
        seq_len = tf.shape(x)[1]
        mask = tf.where(tf.equal(x,0), tf.ones_like(x)*-1e9, tf.zeros_like(x))
        #mask = tf.cast(tf.math.equal(x, 0), tf.float32)
        #mask = mask[:, tf.newaxis, :]
        mask = tf.expand_dims(mask, axis=1)

        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)
        
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, mask, training=training)
        return x, mask


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, num_heads, d_model, dff, target_vocab_size,
            maximum_position_encoding, rate=0.1, name=None):
        super(Decoder, self).__init__(name=name)
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

        self.dec_layers = [DecoderLayer(num_heads, d_model, dff, rate, name=self.name + str(i)) 
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


    def call(self, x, enc_output, enc_mask, training=True):

        seq_len = tf.shape(x)[1]

        look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        dec_target_padding_mask = tf.where(tf.equal(x,0), tf.ones_like(x), tf.zeros_like(x))
        #dec_target_padding_mask = tf.cast(tf.math.equal(x, 0), tf.float32)
        #dec_target_padding_mask = dec_target_padding_mask[:, tf.newaxis, :]
        dec_target_padding_mask = tf.expand_dims(dec_target_padding_mask, axis=1)
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)*-1e9

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)
        
        for i in range(self.num_layers):
            x = self.dec_layers[i](x, enc_output, combined_mask, enc_mask, training=training)
        return x