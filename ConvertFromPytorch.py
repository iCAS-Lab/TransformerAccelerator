#
# Author: Brendan Reidy
# Email: bcreidy@email.sc.edu
# Date created: Sep 11 2022
# Last Modified: Sep 11 2022
#

import tensorflow as tf
import torch
import numpy as np
import json
import TransformerModel
import re
import os
from tensorflow.python.training import py_checkpoint_reader


os.environ["CUDA_VISIBLE_DEVICES"]="1"
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
def load_checkpoint(pytorch_model_path):

    # Need to say "model.ckpt" instead of "model.ckpt.index" for tf v2

    model = torch.load(pytorch_model_path)
    state_dict = model["model"]
    #for key in state_dict:
    #    print(key, state_dict[key].size())

    return state_dict

def from_pytorch_model(pytorch_model_path, configPath, strategy=None):

    tf1_checkpoint = load_checkpoint(pytorch_model_path)
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
                return weight_name, state_dict[weight_name].detach().numpy()
        elif name in weight_name:
            return weight_name, state_dict[weight_name].detach().numpy()
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
            if not weight.shape==inWeight.shape:
                raise Exception("ModelConverter could not inject weight with shape: " + str(inWeight.shape) + " into shape " + str(weight.shape))
            tempName = weight.name
            model.weights[i].assign(inWeight)
            outputMap.append((pseudoName, weight.name))
            removeFromList(weight.name)
            return
    raise Exception("ModelConverter was unable to find layer: " + name + "\nDid you mean " + str(closestVal))
        
"""

encoder.sentence_encoder.emb_layer_norm.weight torch.Size([768])
encoder.sentence_encoder.emb_layer_norm.bias torch.Size([768])
"""
def injectEmbeddings(fromModel, toModel):
    cName, word_embeddings = getWeightByName(fromModel, "embed_tokens.weight")
    setWeightByName(toModel, "word_embeddings", word_embeddings, cName)
    cName, position_embedding = getWeightByName(fromModel, "embed_positions.weight")
    setWeightByName(toModel, "position_embedding", position_embedding, cName)
    #cName, type_embeddings = getWeightByName(fromModel, "type_embeddings")
    #setWeightByName(toModel, "type_embeddings", type_embeddings, cName)

    cName, layer_norm_gamma = getWeightByName(fromModel, "emb_layer_norm.weight")
    setWeightByName(toModel, "transformer/bert_embedding/layer_normalization/gamma", layer_norm_gamma, cName)
    cName, layer_norm_beta = getWeightByName(fromModel, "emb_layer_norm.bias")
    setWeightByName(toModel, "transformer/bert_embedding/layer_normalization/beta", layer_norm_beta, cName)
    print("Successfuly injected embedding values")

def injectMHA(fromModel, toModel, num_heads, layer=0):
    n1, attn_layer_norm_gamma = getWeightByName(fromModel, "layers." + str(layer) + ".self_attn_layer_norm.weight")
    n2, attn_layer_norm_beta = getWeightByName(fromModel, "layers." + str(layer) + ".self_attn_layer_norm.bias")
    n3, out_layer_norm_gamma = getWeightByName(fromModel, "layers." + str(layer) + ".final_layer_norm.weight")
    n4, out_layer_norm_beta = getWeightByName(fromModel, "layers." + str(layer) + ".final_layer_norm.bias")

    setWeightByName(toModel, "layer_" + str(layer) + "/attention_layer_norm/gamma", attn_layer_norm_gamma,n1)
    setWeightByName(toModel, "layer_" + str(layer) + "/attention_layer_norm/beta", attn_layer_norm_beta,n2)
    setWeightByName(toModel, "layer_" + str(layer) + "/output_layer_norm/gamma", out_layer_norm_gamma,n3)
    setWeightByName(toModel, "layer_" + str(layer) + "/output_layer_norm/beta", out_layer_norm_beta,n4)

    
    n1,intermediate_kernel = getWeightByName(fromModel, "layers." + str(layer) + ".fc1.weight")
    intermediate_kernel = np.swapaxes(intermediate_kernel,0,1)
    n2,intermediate_bias = getWeightByName(fromModel, "layers." + str(layer) + ".fc1.bias")
    setWeightByName(toModel, "layer_" + str(layer) + "/intermediate/kernel", intermediate_kernel,n1)
    setWeightByName(toModel, "layer_" + str(layer) + "/intermediate/bias", intermediate_bias,n2)

    n1,output_kernel = getWeightByName(fromModel, "layers." + str(layer) + ".fc2.weight")
    output_kernel = np.swapaxes(intermediate_kernel,0,1)
    n2,output_bias = getWeightByName(fromModel, "layers." + str(layer) + ".fc2.bias")
    setWeightByName(toModel, "layer_" + str(layer) + "/kernel_out:", output_kernel, n1)
    setWeightByName(toModel, "layer_" + str(layer) + "/bias_out:", output_bias, n2)

    n1,query_kernel = getWeightByName(fromModel, "layers." + str(layer) + ".self_attn.q_proj.weight")
    n2,query_bias = getWeightByName(fromModel, "layers." + str(layer) + ".self_attn.q_proj.bias")
    n3,key_kernel = getWeightByName(fromModel, "layers." + str(layer) + ".self_attn.k_proj.weight")
    n4,key_bias = getWeightByName(fromModel, "layers." + str(layer) + ".self_attn.k_proj.bias")
    n5,value_kernel = getWeightByName(fromModel, "layers." + str(layer) + ".self_attn.v_proj.weight")
    n6,value_bias = getWeightByName(fromModel, "layers." + str(layer) + ".self_attn.v_proj.bias")
    attn_output_kernel_name,attn_output_kernel = getWeightByName(fromModel, "layers." + str(layer) + ".self_attn.out_proj.weight")
    n7,attn_output_bias = getWeightByName(fromModel, "layers." + str(layer) + ".self_attn.out_proj.bias")

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

    """
    n1,pooler_kernel = getWeightByName(fromModel, "pooler/dense/kernel")
    n2,pooler_bias = getWeightByName(fromModel, "pooler/dense/bias")
    setWeightByName(toModel, "transformer/transformerpooler_transform_kernel", pooler_kernel,n1)
    setWeightByName(toModel, "transformer/transformerpooler_transform_bias", pooler_bias,n2)
    """
    showOuputMap(outdir="model_map_pt.log")

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
            fp.write("*"*25 + "\n")
            for n in unusedValues.keys():
                if unusedValues[n] is None:
                    continue
                fp.write(str(n) + " " + str(unusedValues[n].shape) + "\n")
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


PATH = "/home/brendan/IBERT/I-BERT/outputs/none/QQP-base/wd0.1_ad0.1_d0.1_lr1e-5/0913-160322_ckpt/checkpoint_best.pt"
config_file = "/home/brendan/TransformerAccelerator/models/roberta_config.json"
from_pytorch_model(PATH, config_file)