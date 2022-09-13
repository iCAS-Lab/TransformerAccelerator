#
# Author: Brendan Reidy
# Email: bcreidy@email.sc.edu
# Date created: Sep 10 2022
# Last Modified: Sep 11 2022
#

import tensorflow as tf
import json
import TransformerModel
import re
WORD = re.compile(r"\w+")
mappedWeights = []
outputMap = []
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


def from_hub_encoder(hub_encoder, configPath, strategy=None):
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

    with strategy.scope():
        x = tf.keras.layers.Input(shape=(128), dtype=tf.float32, name="input_word_ids", ragged=False)
        seg = tf.keras.layers.Input(shape=(128), dtype=tf.float32, name="input_type_ids", ragged=False)
        mask = tf.keras.layers.Input(shape=(128), dtype=tf.float32, name="input_mask", ragged=False)
        custom_encoder = TransformerModel.BERT(n_layers, num_heads, vocab_size, seq_len, n_segments, d_model, intermediate_size, activation=activation, name="transformer")(x, seg)
        encoder_model = tf.keras.Model(inputs=[x, seg, mask], outputs=[custom_encoder])
    inject_weights(hub_encoder, encoder_model, n_layers, num_heads)
    return encoder_model

def removeFromList(name):
    global mappedWeights
    if name in mappedWeights:
        mappedWeights.remove(name)
    else:
        print("ERROR", name, "not in list")
def getWeightByName(model, name, exact=False):
    for weight in model.weights:
        if exact:
            if weight.name == name:
                return weight
        elif name in weight.name:
            return weight
    return None

def setWeightByName(model, name, inWeight, pseudoName=None):
    global outputMap
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
            if pseudoName is None:
                outputMap.append((inWeight.name, weight.name))
            else:
                outputMap.append((pseudoName, weight.name))
            removeFromList(weight.name)
            return
    raise Exception("ModelConverter was unable to find layer: " + name + "\nDid you mean " + str(closestVal))
        

def injectEmbeddings(fromModel, toModel):
    word_embeddings = getWeightByName(fromModel, "word_embeddings")
    setWeightByName(toModel, "word_embeddings", word_embeddings)
    position_embedding = getWeightByName(fromModel, "position_embedding")
    setWeightByName(toModel, "position_embedding", position_embedding)
    type_embeddings = getWeightByName(fromModel, "type_embeddings")
    setWeightByName(toModel, "type_embeddings", type_embeddings)

    layer_norm_gamma = getWeightByName(fromModel, "embeddings/layer_norm/gamma")
    setWeightByName(toModel, "transformer/bert_embedding/layer_normalization/gamma", layer_norm_gamma)
    layer_norm_beta = getWeightByName(fromModel, "embeddings/layer_norm/beta")
    setWeightByName(toModel, "transformer/bert_embedding/layer_normalization/beta", layer_norm_beta)
    print("Successfuly injected embedding values")

def injectMHA(fromModel, toModel, num_heads, layer=0):
    attn_layer_norm_gamma = getWeightByName(fromModel, "transformer/layer_" + str(layer) + "/self_attention_layer_norm/gamma")
    attn_layer_norm_beta = getWeightByName(fromModel, "transformer/layer_" + str(layer) + "/self_attention_layer_norm/beta")
    out_layer_norm_gamma = getWeightByName(fromModel, "transformer/layer_" + str(layer) + "/output_layer_norm/gamma")
    out_layer_norm_beta = getWeightByName(fromModel, "transformer/layer_" + str(layer) + "/output_layer_norm/beta")
    setWeightByName(toModel, "layer_" + str(layer) + "/attention_layer_norm/gamma", attn_layer_norm_gamma)
    setWeightByName(toModel, "layer_" + str(layer) + "/attention_layer_norm/beta", attn_layer_norm_beta)
    setWeightByName(toModel, "layer_" + str(layer) + "/output_layer_norm/gamma", out_layer_norm_gamma)
    setWeightByName(toModel, "layer_" + str(layer) + "/output_layer_norm/beta", out_layer_norm_beta)

    intermediate_kernel = getWeightByName(fromModel, "transformer/layer_" + str(layer) + "/intermediate/kernel")
    intermediate_bias = getWeightByName(fromModel, "transformer/layer_" + str(layer) + "/intermediate/bias")
    setWeightByName(toModel, "layer_" + str(layer) + "/intermediate/kernel", intermediate_kernel)
    setWeightByName(toModel, "layer_" + str(layer) + "/intermediate/bias", intermediate_bias)

    output_kernel = getWeightByName(fromModel, "transformer/layer_" + str(layer) + "/output/kernel")
    output_bias = getWeightByName(fromModel, "transformer/layer_" + str(layer) + "/output/bias")
    setWeightByName(toModel, "layer_" + str(layer) + "/kernel_out:", output_kernel)
    setWeightByName(toModel, "layer_" + str(layer) + "/bias_out:", output_bias)

    query_kernel = getWeightByName(fromModel, "transformer/layer_" + str(layer) + "/self_attention/query/kernel")
    query_bias = getWeightByName(fromModel, "transformer/layer_" + str(layer) + "/self_attention/query/bias")
    key_kernel = getWeightByName(fromModel, "transformer/layer_" + str(layer) + "/self_attention/key/kernel")
    key_bias = getWeightByName(fromModel, "transformer/layer_" + str(layer) + "/self_attention/key/bias")
    value_kernel = getWeightByName(fromModel, "transformer/layer_" + str(layer) + "/self_attention/value/kernel")
    value_bias = getWeightByName(fromModel, "transformer/layer_" + str(layer) + "/self_attention/value/bias")
    attn_output_kernel = getWeightByName(fromModel, "transformer/layer_" + str(layer) + "/self_attention/attention_output/kernel")
    attn_output_kernel_name = attn_output_kernel.name
    attn_output_bias = getWeightByName(fromModel, "transformer/layer_" + str(layer) + "/self_attention/attention_output/bias")

    attn_output_kernel = tf.reshape(attn_output_kernel, (attn_output_kernel.shape[1]*num_heads, attn_output_kernel.shape[2]))
    setWeightByName(toModel, "layer_" + str(layer) + "/mha/mhaattention_output_kernel:", attn_output_kernel, pseudoName=attn_output_kernel_name)
    setWeightByName(toModel, "layer_" + str(layer) + "/mha/mhaattention_output_bias:", attn_output_bias)

    for h in range(num_heads):
        queryTempK = query_kernel[:,h:h+1,:]
        queryTempK = tf.reshape(queryTempK, (queryTempK.shape[0], queryTempK.shape[2]))
        queryTempB = query_bias[h:h+1,:]
        queryTempB = tf.reshape(queryTempB, (queryTempB.shape[1]))

        keyTempK = key_kernel[:,h:h+1,:]
        keyTempK = tf.reshape(keyTempK, (keyTempK.shape[0], keyTempK.shape[2]))
        keyTempB = query_bias[h:h+1,:]
        keyTempB = tf.reshape(keyTempB, (keyTempB.shape[1]))

        valueTempK = value_kernel[:,h:h+1,:]
        valueTempK = tf.reshape(valueTempK, (valueTempK.shape[0], valueTempK.shape[2]))
        valueTempB = value_bias[h:h+1,:]
        valueTempB = tf.reshape(valueTempB, (valueTempB.shape[1]))

        setWeightByName(toModel, "layer_" + str(layer) + "/mha/mhasdp_" + str(h) + "/mhasdp_" + str(h) + "_kernel_q", queryTempK, pseudoName=query_kernel.name)
        setWeightByName(toModel, "layer_" + str(layer) + "/mha/mhasdp_" + str(h) + "/mhasdp_" + str(h) + "_bias_q", queryTempB, pseudoName=query_bias.name)

        setWeightByName(toModel, "layer_" + str(layer) + "/mha/mhasdp_" + str(h) + "/mhasdp_" + str(h) + "_kernel_k", keyTempK, pseudoName=key_kernel.name)
        setWeightByName(toModel, "layer_" + str(layer) + "/mha/mhasdp_" + str(h) + "/mhasdp_" + str(h) + "_bias_k", keyTempB, pseudoName=key_kernel.name)

        setWeightByName(toModel, "layer_" + str(layer) + "/mha/mhasdp_" + str(h) + "/mhasdp_" + str(h) + "_kernel_v", valueTempK, pseudoName=value_kernel.name)
        setWeightByName(toModel, "layer_" + str(layer) + "/mha/mhasdp_" + str(h) + "/mhasdp_" + str(h) + "_bias_v", valueTempB, pseudoName=value_bias.name)





    


def inject_weights(fromModel, toModel, n_layers, num_heads):
    global mappedWeights
    global outputMap
    mappedWeights = []
    for weight in toModel.weights:
        mappedWeights.append(weight.name)
    injectEmbeddings(fromModel, toModel)
    for layer in range(n_layers):
        injectMHA(fromModel, toModel, num_heads, layer=layer)

    pooler_kernel = getWeightByName(fromModel, "pooler_transform/kernel")
    pooler_bias = getWeightByName(fromModel, "pooler_transform/bias")
    setWeightByName(toModel, "transformer/transformerpooler_transform_kernel", pooler_kernel)
    setWeightByName(toModel, "transformer/transformerpooler_transform_bias", pooler_bias)
    showOuputMap(outdir=toModel.name + "_mapping.log")

def showOuputMap(outdir=None):
    global outputMap
    global mappedWeights
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