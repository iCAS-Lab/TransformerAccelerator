#
# Author: Brendan Reidy
# Email: bcreidy@email.sc.edu
# Date created: Sep 11 2022
# Last Modified: Sep 11 2022
#

import tensorflow as tf
import json
import TransformerModel
import re
from tensorflow.python.training import py_checkpoint_reader
WORD = re.compile(r"\w+")
mappedWeights = []
outputMap = []
unusedValues = {}
def word2vec(word):
    from collections import Counter
    from math import sqrt

    # count the characters in word
    cw = Counter(word)
    # precomputes a set of the different characters
    sw = set(cw)
    # precomputes the "length" of the word vector
    lw = sqrt(sum(c*c for c in cw.values()))

    # return a tuple
    return cw, sw, lw

def cosdis(v1, v2):
    v1 = word2vec(v1)
    v2 = word2vec(v2)
    # which characters are common to the two words?
    common = v1[1].intersection(v2[1])
    # by definition of cosine distance we have
    return sum(v1[0][ch]*v2[0][ch] for ch in common)/v1[2]/v2[2]
def load_checkpoint(file_name):

    # Need to say "model.ckpt" instead of "model.ckpt.index" for tf v2
    reader = py_checkpoint_reader.NewCheckpointReader(file_name)

    state_dict = {
        v: reader.get_tensor(v) for v in reader.get_variable_to_shape_map()
    }
    return state_dict

def from_tf1_checkpoint(tf1_checkpoint_path, configPath, strategy=None):

    tf1_checkpoint = load_checkpoint(tf1_checkpoint_path)
    global unusedValues
    unusedValues = tf1_checkpoint

    with open(configPath) as json_file:
        data = json.load(json_file)
    n_layers = data["num_hidden_layers"]
    num_heads = data["num_attention_heads"]
    vocab_size = data["vocab_size"]
    seq_len = data["max_position_embeddings"]
    n_segments = data["type_vocab_size"]
    d_model = data["hidden_size"]
    intermediate_size = data["intermediate_size"]
    activation = data["hidden_act"]

    x = tf.keras.layers.Input(shape=(128), dtype=tf.float32, name="input_word_ids", ragged=False)
    seg = tf.keras.layers.Input(shape=(128), dtype=tf.float32, name="input_type_ids", ragged=False)
    mask = tf.keras.layers.Input(shape=(128), dtype=tf.float32, name="input_mask", ragged=False)
    custom_encoder = TransformerModel.BERT(n_layers, num_heads, vocab_size, seq_len, n_segments, d_model, intermediate_size, activation=activation, name="transformer")(x, seg)
    encoder_model = tf.keras.Model(inputs=[x, seg, mask], outputs=[custom_encoder])
    encoder_model.compile()
    inject_weights(tf1_checkpoint, encoder_model, n_layers, num_heads)
    return encoder_model

from tensorflow.keras.models import Model
def BERT_Classifier(backbone_model, classes, strategy = None):
    backbone = backbone_model
    x = backbone.output
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(classes, activation='tanh')(x)
    model = Model(inputs=backbone.input, outputs=x)
    return model

def removeFromList(name):
    global mappedWeights
    if name in mappedWeights:
        mappedWeights.remove(name)
    else:
        print("ERROR", name, "not in list")
def getWeightByName(state_dict, name, exact=False):
    closest = -9999999999999999999
    closestVal = None
    for weight_name in state_dict.keys():
        sim = cosdis(name, weight_name)
        if sim > closest:
            closest = sim
            closestVal = weight_name
        if exact:
            if weight_name == name:
                return weight_name, state_dict[weight_name]
        elif name in weight_name:
            return weight_name, state_dict[weight_name]
    raise Exception("ModelConverter was unable to find layer: " + name + "\nDid you mean " + str(closestVal))
    return None

def setWeightByName(model, name, inWeight, pseudoName):
    global outputMap
    global unusedValues
    unusedValues[pseudoName] = None
    closest = -9999999999999999999
    closestVal = None
    for i, weight in enumerate(model.weights):
        sim = cosdis(name, weight.name)
        if sim > closest:
            closest = sim
            closestVal = weight.name
        if name in weight.name:
            assert weight.shape==inWeight.shape
            tempName = weight.name
            model.weights[i].assign(inWeight)
            outputMap.append((pseudoName, weight.name))
            removeFromList(weight.name)
            return
    raise Exception("ModelConverter was unable to find layer: " + name + "\nDid you mean " + str(closestVal))
        

def injectEmbeddings(fromModel, toModel):
    cName, word_embeddings = getWeightByName(fromModel, "word_embeddings")
    setWeightByName(toModel, "word_embeddings", word_embeddings, cName)
    cName, position_embedding = getWeightByName(fromModel, "position_embedding")
    setWeightByName(toModel, "position_embedding", position_embedding, cName)
    cName, type_embeddings = getWeightByName(fromModel, "type_embeddings")
    setWeightByName(toModel, "type_embeddings", type_embeddings, cName)

    cName, layer_norm_gamma = getWeightByName(fromModel, "embeddings/LayerNorm/gamma")
    setWeightByName(toModel, "transformer/bert_embedding/layer_normalization/gamma", layer_norm_gamma, cName)
    cName, layer_norm_beta = getWeightByName(fromModel, "embeddings/LayerNorm/beta")
    setWeightByName(toModel, "transformer/bert_embedding/layer_normalization/beta", layer_norm_beta, cName)
    print("Successfuly injected embedding values")

def injectMHA(fromModel, toModel, num_heads, layer=0):
    n1, attn_layer_norm_gamma = getWeightByName(fromModel, "layer_" + str(layer) + "/attention/output/LayerNorm/gamma")
    n2, attn_layer_norm_beta = getWeightByName(fromModel, "layer_" + str(layer) + "/attention/output/LayerNorm/beta")
    n3, out_layer_norm_gamma = getWeightByName(fromModel, "layer_" + str(layer) + "/output/LayerNorm/gamma")
    n4, out_layer_norm_beta = getWeightByName(fromModel, "layer_" + str(layer) + "/output/LayerNorm/beta")
    setWeightByName(toModel, "layer_" + str(layer) + "/attention_layer_norm/gamma", attn_layer_norm_gamma,n1)
    setWeightByName(toModel, "layer_" + str(layer) + "/attention_layer_norm/beta", attn_layer_norm_beta,n2)
    setWeightByName(toModel, "layer_" + str(layer) + "/output_layer_norm/gamma", out_layer_norm_gamma,n3)
    setWeightByName(toModel, "layer_" + str(layer) + "/output_layer_norm/beta", out_layer_norm_beta,n4)

    n1,intermediate_kernel = getWeightByName(fromModel, "layer_" + str(layer) + "/intermediate/dense/kernel")
    n2,intermediate_bias = getWeightByName(fromModel, "layer_" + str(layer) + "/intermediate/dense/bias")
    setWeightByName(toModel, "layer_" + str(layer) + "/intermediate/kernel", intermediate_kernel,n1)
    setWeightByName(toModel, "layer_" + str(layer) + "/intermediate/bias", intermediate_bias,n2)
    

    n1,output_kernel = getWeightByName(fromModel, "layer_" + str(layer) + "/output/dense/kernel")
    n2,output_bias = getWeightByName(fromModel, "layer_" + str(layer) + "/output/dense/bias")
    setWeightByName(toModel, "layer_" + str(layer) + "/kernel_out:", output_kernel, n1)
    setWeightByName(toModel, "layer_" + str(layer) + "/bias_out:", output_bias, n2)

    n1,query_kernel = getWeightByName(fromModel, "layer_" + str(layer) + "/attention/self/query/kernel")
    n2,query_bias = getWeightByName(fromModel, "layer_" + str(layer) + "/attention/self/query/bias")
    n3,key_kernel = getWeightByName(fromModel, "layer_" + str(layer) + "/attention/self/key/kernel")
    n4,key_bias = getWeightByName(fromModel, "layer_" + str(layer) + "/attention/self/key/bias")
    n5,value_kernel = getWeightByName(fromModel, "layer_" + str(layer) + "/attention/self/value/kernel")
    n6,value_bias = getWeightByName(fromModel, "layer_" + str(layer) + "/attention/self/value/bias")
    attn_output_kernel_name,attn_output_kernel = getWeightByName(fromModel, "layer_" + str(layer) + "/attention/output/dense/kernel")
    n7,attn_output_bias = getWeightByName(fromModel, "layer_" + str(layer) + "/attention/output/dense/bias")

    setWeightByName(toModel, "layer_" + str(layer) + "/mha/mhaattention_output_kernel:", attn_output_kernel, attn_output_kernel_name)
    setWeightByName(toModel, "layer_" + str(layer) + "/mha/mhaattention_output_bias:", attn_output_bias,n7)
    
    d_size = int(query_kernel.shape[0] / num_heads)
    for h in range(num_heads):
        queryTempK = query_kernel[:,h*d_size:(h+1)*d_size]
        queryTempB = query_bias[h*d_size:(h+1)*d_size]

        keyTempK = key_kernel[:,h*d_size:(h+1)*d_size]
        keyTempB = key_bias[h*d_size:(h+1)*d_size]

        valueTempK = value_kernel[:,h*d_size:(h+1)*d_size]
        valueTempB = value_bias[h*d_size:(h+1)*d_size]

        setWeightByName(toModel, "layer_" + str(layer) + "/mha/mhasdp_" + str(h) + "/mhasdp_" + str(h) + "_kernel_q", queryTempK, n1)
        setWeightByName(toModel, "layer_" + str(layer) + "/mha/mhasdp_" + str(h) + "/mhasdp_" + str(h) + "_bias_q", queryTempB, n2)

        setWeightByName(toModel, "layer_" + str(layer) + "/mha/mhasdp_" + str(h) + "/mhasdp_" + str(h) + "_kernel_k", keyTempK, n3)
        setWeightByName(toModel, "layer_" + str(layer) + "/mha/mhasdp_" + str(h) + "/mhasdp_" + str(h) + "_bias_k", keyTempB, n4)

        setWeightByName(toModel, "layer_" + str(layer) + "/mha/mhasdp_" + str(h) + "/mhasdp_" + str(h) + "_kernel_v", valueTempK, n5)
        setWeightByName(toModel, "layer_" + str(layer) + "/mha/mhasdp_" + str(h) + "/mhasdp_" + str(h) + "_bias_v", valueTempB, n6)
    #"""




    


def inject_weights(fromModel, toModel, n_layers, num_heads):
    global mappedWeights
    global outputMap
    mappedWeights = []
    for weight in toModel.weights:
        mappedWeights.append(weight.name)
    injectEmbeddings(fromModel, toModel)
    for layer in range(n_layers):
       injectMHA(fromModel, toModel, num_heads, layer=layer)

    n1,pooler_kernel = getWeightByName(fromModel, "pooler/dense/kernel")
    n2,pooler_bias = getWeightByName(fromModel, "pooler/dense/bias")
    setWeightByName(toModel, "transformer/transformerpooler_transform_kernel", pooler_kernel,n1)
    setWeightByName(toModel, "transformer/transformerpooler_transform_bias", pooler_bias,n2)
    showOuputMap(outdir="model_mapping.log")

def showOuputMap(outdir=None):
    global outputMap
    global mappedWeights
    global unusedValues
    if not outdir is None:
        with open(outdir, 'w') as fp:
            for b,a in outputMap:
                fp.write("[X] " +  a + " -> " + b + "\n")
            for a in mappedWeights:
                fp.write("[_] " +  a + " -> \n")
    else:
        for b,a in outputMap:
            print("[X]", a, "->", b)
        for a in mappedWeights:
            print("[_]", a, "->")

        print("*"*25)
        for n in unusedValues.keys():
            if unusedValues[n] is None:
                continue
            print(n, unusedValues[n].shape)



def getTrainableParams(model):
    totalSize = 0
    for weight in model.weights:
        currSize = 1
        if "Variable:0" in weight.name:
            continue
        for axis in weight.shape:
            currSize*=axis
        totalSize+=currSize
    return totalSize