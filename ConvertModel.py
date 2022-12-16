#
# Author: Brendan Reidy
# Email: bcreidy@email.sc.edu
# Date created: Sep 10 2022
# Last Modified: Dec 12 2022
#

import ConvertFromTF1Checkpoint
import os
import tensorflow as tf
import json
import TransformerModel
import re
from tensorflow.python.training import py_checkpoint_reader
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

def BERT_Classifier(backbone_model, classes, use_conv=False):
    backbone = backbone_model
    x = backbone.output
    x = tf.keras.layers.Dropout(0.1)(x)
    x = TransformerModel.ConfigurableDense(classes, inp_size=backbone.output.shape[-1], use_conv=use_conv, activation='tanh')(x)
    model = tf.keras.Model(inputs=backbone.input, outputs=x)
    return model

def clone_from_archtype(model, configPath, archtype=TransformerModel, partition_config = None):
    if partition_config==None:
        partition_config=archtype.DEFAULT_PARTITION_CONFIG
    
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
    custom_encoder = archtype.BERT(n_layers, num_heads, vocab_size, seq_len, n_segments, d_model, intermediate_size,
        activation=activation, partition_config=partition_config, name="transformer")(x, seg, mask)
    encoder_model = tf.keras.Model(inputs=[x, seg, mask], outputs=[custom_encoder])
    classifer = BERT_Classifier(encoder_model, 2)
    classifer = clone_to_new(model, classifer)
    return classifer


def clone_to_new(modelA, modelB):
    # clones the weights of one model onto another model
    for i, inWeight in enumerate(modelA.weights):
        name = inWeight.name
        closest = -9999999999999999999
        closestWeight = None
        closestIdx = -1
        for j, weight in enumerate(modelB.weights):
            sim = cosdis(name, weight.name)

            # for some reason cosine distance doesnt care about the numbers so we need to verify manually...
            # TODO find some way to do this more generically
            if "embeddings:0" in name:
                if not "embeddings:0" in weight.name:
                    continue
                emb_type1 = name.split("/")[2]
                emb_type2 = weight.name.split("/")[2]
                if not emb_type1==emb_type2:
                    continue
            if "layer" in name and not "layer_n" in name:
                if not "layer" in weight.name:
                    continue
                layer_num1 = name.split("/")[1]
                layer_num2 = weight.name.split("/")[1]
                if not layer_num1==layer_num2:
                    continue
            if "mhasdp" in name:
                if not "mhasdp" in weight.name:
                    continue
                mha_num1 = name.split("/")[3]
                mha_num2 = weight.name.split("/")[3]
                if not mha_num1==mha_num2:
                    continue

            if sim > closest:
                closest = sim
                closestWeight = weight
                closestIdx = j
        
        if closestIdx==-1:
            #print("WARNING: No weight map found for:", name)
            continue
        if not closestWeight.shape==inWeight.shape:
            #print("WARNING: Shape mismatch for:")
            #print("     ", name, closestWeight.name)
            continue
        #print(name, "-->", closestWeight.name)
        modelB.weights[closestIdx].assign(inWeight.numpy())
    return modelB

def from_config(configPath, partition_config = None):
    if partition_config==None:
        partition_config=TransformerModel.DEFAULT_PARTITION_CONFIG
    
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
    custom_encoder = TransformerModel.BERT(n_layers, num_heads, vocab_size, seq_len, n_segments, d_model, intermediate_size,
        activation=activation, partition_config=partition_config, name="transformer")(x, seg, mask)
    encoder_model = tf.keras.Model(inputs=[x, seg, mask], outputs=[custom_encoder])
    return encoder_model

# TODO depricate fub encoder
#def from_hub_encoder(hub_encoder, configPath, strategy=None):
#    return ConvertModelFromHub.from_hub_encoder(hub_encoder, configPath, strategy=strategy)

def from_tf1_checkpoint(model_dir, partition_config=None):
    if partition_config==None:
        partition_config=TransformerModel.DEFAULT_PARTITION_CONFIG
    return ConvertFromTF1Checkpoint.from_tf1_checkpoint(
        os.path.join(model_dir, "bert_model.ckpt"),
        os.path.join(model_dir, "bert_config.json"),
        partition_config
    )
