from tensorflow.keras.models import Model
import ConvertModelFromHub
import ConvertFromTF1Checkpoint
import os
import tensorflow as tf

def BERT_Classifier(backbone_model, classes, strategy = None):
    with strategy.scope():
        backbone = backbone_model
        x = backbone.output
        x = tf.keras.layers.Dropout(0.1)(x)
        x = tf.keras.layers.Dense(classes, activation='tanh')(x)
        model = Model(inputs=backbone.input, outputs=x)
    return model

def from_hub_encoder(hub_encoder, configPath, strategy=None):
    return ConvertModelFromHub.from_hub_encoder(hub_encoder, configPath, strategy=strategy)

def from_tf1_checkpoint(model_dir):
    return ConvertFromTF1Checkpoint.from_tf1_checkpoint(
        os.path.join(model_dir, "bert_model.ckpt"),
        os.path.join(model_dir, "bert_config.json")
    )
