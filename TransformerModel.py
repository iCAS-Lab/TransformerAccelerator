#
# Author: Brendan Reidy
# Email: bcreidy@email.sc.edu
# Date created: May 5 2020
# Last Modified: Sep 11 2022
#

import tensorflow as tf
import sys
import numpy as np
import MaskUtils
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
    def __init__(self, d_model, look_back=1, name=None):
        super(ScaledConvolutionalDotProduct, self).__init__(name=name)
        self.d_model = d_model
        self.relu1 = activations.get('relu')
        self.relu2 = activations.get('relu')
        self.relu3 = activations.get('relu')
        self.softmax = activations.get('softmax')
        self.rank = 2
        self.look_back = look_back
        self.strides = conv_utils.normalize_tuple(1, self.rank, 'strides')

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        kernel_shape = (1,self.look_back,input_shape[-1],self.d_model)
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
            name='bias',
            shape=(self.d_model,),
            initializer='random_normal',
            trainable=True,
            dtype=self.dtype)

        self.bias_k = self.add_weight(
            name='bias',
            shape=(self.d_model,),
            initializer='random_normal',
            trainable=True,
            dtype=self.dtype)

        self.bias_v = self.add_weight(
            name='bias',
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
        config = {
            'd_model':self.d_model,
            'look_back':self.look_back,
            'name':self.name
        }
        return config
        #return {'d_model': self.d_model, 'name':self.name, 'rank':self.rank, 'look_back':self.look_back, 'strides':self.strides, 'data_format':self.data_format}

    def conv_matmul(self, a, b, transpose_b=False):
        if a.shape[0] is None:
            return tf.matmul(a,b,transpose_b=transpose_b)

        
        if transpose_b:
            b = tf.transpose(b, [0,2,1])
        
        retVals = []
        for x in range(a.shape[0]):
            a_slice = a[x:x+1]
            a_slice = tf.expand_dims(a_slice, axis=0)
            b_slice = b[x:x+1]
            b_slice = tf.expand_dims(b_slice, axis=0)
            result = self.relu1(self._convolution_op(a_slice,b_slice))
            retVals.append(result)
        concat = tf.concat(retVals, axis=1)
        concat = tf.squeeze(concat,axis=0)
        return concat

    def call(self, q, k, v, mask):
        q = tf.expand_dims(q, axis=0)
        k = tf.expand_dims(k, axis=0)
        v = tf.expand_dims(v, axis=0)

        q = self.relu1(self._convolution_op(q, self.kernel_q) + self.bias_q)
        k = self.relu2(self._convolution_op(k, self.kernel_k) + self.bias_k)
        v = self.relu3(self._convolution_op(v, self.kernel_v) + self.bias_v)
        
        q = tf.squeeze(q,axis=0)
        k = tf.squeeze(k,axis=0)
        v = tf.squeeze(v,axis=0)
        matmul_qk = self.relu1(tf.matmul(q,k, transpose_b=True))
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        if not mask is None:
            scaled_attention_logits += mask
        attention_weights = scaled_attention_logits#self.softmax(scaled_attention_logits, axis=-1)
        output = self.relu1(tf.matmul(attention_weights,v))
        return output

class ScaledDotProduct(tf.keras.layers.Layer):
    def __init__(self, inp_size, d_model, activation='gelu', name=None):
        super(ScaledDotProduct, self).__init__(name=name)
        self.d_model = d_model
        self.activation = activation
        self.inp_size = inp_size
        self.query_activation = activations.get(activation)
        self.key_activation = activations.get(activation)
        self.value_activation = activations.get(activation)
        self.softmax = activations.get('softmax')

    def build(self, input_shape):
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

    def get_config(self):
        return {'d_model': self.d_model, 'name':self.name, 'activation':self.activation, 'inp_size':self.inp_size}


    def call(self, q, k, v, mask=None):
        q = tf.matmul(q, self.kernel_q) + self.bias_q
        k = tf.matmul(k, self.kernel_k) + self.bias_k
        v = tf.matmul(v, self.kernel_v) + self.bias_v

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
    def __init__(self, num_heads, d_model, rate=0.1, activation='gelu', name=None):
        super(BERTMultiHeadedAttention, self).__init__(name=name)
        self.attention_heads = []
        for i in range(num_heads):
            sdp = ScaledDotProduct(d_model, int(d_model/num_heads), name=self.name + 'sdp_' + str(i))
            self.attention_heads.append(sdp)
        self.num_heads = num_heads
        self.rate = rate
        self.activation = activation
        self.act_out = activations.get(activation)
        self.d_model = d_model
        self.dropout = tf.keras.layers.Dropout(rate)

    def build(self, input_shape):
        #for head in self.attention_heads:
            #head.build(input_shape)
        self.kernel = self.add_weight(self.name + "attention_output_kernel",shape=[input_shape[-1],self.d_model],
                initializer='random_normal',
                trainable=True)
        self.bias = self.add_weight(self.name + "attention_output_bias",shape=[self.d_model],
                initializer='random_normal',
                trainable=True)

    def get_config(self):
        return {'d_model': self.d_model, 'num_heads': self.num_heads, 'name':self.name, 'activation':self.activation, 'rate':self.rate}

    def call(self, q, k, v, mask, training=True):
        attention_outputs = []
        for head in self.attention_heads:
            attention_outputs.append(head(q, k, v, mask))
        x = attention_outputs[0]
        if self.num_heads>1:
            x = tf.keras.layers.concatenate(attention_outputs)
        x = tf.matmul(x, self.kernel) + self.bias
        x = self.dropout(x, training=training)
        return x

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
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-12)

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
    def __init__(self, n_layers, num_heads, vocab_size, seq_len, n_segments, d_model, intermediate_size, rate=0.1, activation='gelu', name=None):
        super(BERT, self).__init__(name=name)
        self.n_layers = n_layers
        self.num_heads = num_heads
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.n_segments = n_segments
        self.d_model = d_model
        self.intermediate_size = intermediate_size


        self.embedding = BertEmbedding(vocab_size, seq_len, n_segments, d_model)
        self.enc_layers = [BertEncoder(num_heads, d_model, intermediate_size, rate, activation=activation, name="layer_" + str(i)) 
                        for i in range(n_layers)]

        self.activation = activation
        self.act_out = activations.get(activation)

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
            'activation':self.activation
            }
    def build(self, input_shape):
        self.pooler_transform = self.add_weight(self.name + "pooler_transform_kernel",shape=[self.d_model,self.d_model],
                initializer='random_normal',
                trainable=True)
        self.pooler_transform_bias = self.add_weight(self.name + "pooler_transform_bias",shape=[self.d_model],
                initializer='random_normal',
                trainable=True)

    def call(self, x, seg, training=True):
        mask = tf.where(tf.equal(x,0), tf.ones_like(x)*-1e9, tf.zeros_like(x))
        mask = tf.expand_dims(mask, axis=1)

        x = self.embedding(x,seg)
        for layer in self.enc_layers:
            x = layer(x, mask)
        x = x[:,0]
        x = self.act_out(tf.matmul(x, self.pooler_transform) + self.pooler_transform_bias)
        return x
        #return {"pooled_output":x}

class BertEncoder(tf.keras.layers.Layer):
    def __init__(self, num_heads, d_model, intermediate_size, rate=0.1, activation='gelu', name=None):
        super(BertEncoder, self).__init__(name=name)

        self.d_model = d_model
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.rate = rate

        self.mha = BERTMultiHeadedAttention(num_heads, d_model, activation=activation, name = "mha")

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-12, name="attention_layer_norm")
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-12, name="output_layer_norm")
        
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.activation = activation

        self.activation1 = activations.get(activation)
        self.activation2 = activations.get(activation)

    def get_config(self):
        return {
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'intermediate_size':self.intermediate_size,
            'rate':self.rate,
            'name':self.name,
            'activation':self.activation
            }

    def build(self, input_shape):

        #self.mha.build(input_shape)

        self.kernel_dff = self.add_weight(self.name + "/intermediate/kernel",shape=[self.d_model,self.intermediate_size],
                initializer='random_normal',
                trainable=True)
        self.bias_dff = self.add_weight(self.name + "/intermediate/bias",shape=[self.intermediate_size],
                initializer='random_normal',
                trainable=True)

        self.kernel_out = self.add_weight(self.name + "/kernel_out",shape=[self.intermediate_size,self.d_model],
                initializer='random_normal',
                trainable=True)
        self.bias_out = self.add_weight(self.name + "/bias_out",shape=[self.d_model],
                initializer='random_normal',
                trainable=True)

    def call(self, x, mask, training=True):
        #x = tf.squeeze(x)
        attn_output = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        ffn_output1 = self.activation1(tf.matmul(out1, self.kernel_dff) + self.bias_dff)
        ffn_output2 = tf.matmul(ffn_output1, self.kernel_out) + self.bias_out
        ffn_output3 = self.dropout2(ffn_output2, training=training)
        out2 = self.layernorm2(out1 + ffn_output3)
        return out2



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
        #x = tf.squeeze(x)
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