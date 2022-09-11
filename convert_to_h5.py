import os

import sys
  
if len(sys.argv) < 2:
    print("usage: python", sys.argv[0], "<model_directory>")
    exit()

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_models as tfm
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import TransformerModel

import json
import ConvertModel
model_name = sys.argv[1]


#strategy = tf.distribute.MirroredStrategy()#devices=["/gpu:0", "/gpu:1", "/gpu:2"])
strategy = tf.distribute.get_strategy()
BATCH_SIZE_PER_REPLICA = 32
batch_size = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

glue, info = tfds.load('glue/mrpc',
                       with_info=True,
                       batch_size=batch_size)

tokenizer = tfm.nlp.layers.FastWordpieceBertTokenizer(
    vocab_file=os.path.join(model_name, "vocab.txt"),
    lower_case=True)

max_seq_length = 128

packer = tfm.nlp.layers.BertPackInputs(
    seq_length=max_seq_length,
    special_tokens_dict = tokenizer.get_special_tokens_dict())

class BertInputProcessor(tf.keras.layers.Layer):
  def __init__(self, tokenizer, packer):
    super().__init__()
    self.tokenizer = tokenizer
    self.packer = packer

  def call(self, inputs):
    tok1 = self.tokenizer(inputs['sentence1'])
    tok2 = self.tokenizer(inputs['sentence2'])

    packed = self.packer([tok1, tok2])

    if 'label' in inputs:
      return packed, inputs['label']
    else:
      return packed

bert_inputs_processor = BertInputProcessor(tokenizer, packer)
glue_train = glue['train'].map(bert_inputs_processor).prefetch(1)
glue_validation = glue['validation'].map(bert_inputs_processor).prefetch(1)
glue_test = glue['test'].map(bert_inputs_processor).prefetch(1)

bert_encoder = ConvertModel.from_tf1_checkpoint(model_name)
bert_classifier = ConvertModel.BERT_Classifier(bert_encoder, 2, strategy = strategy)
bert_classifier.summary()

epochs = 10
eval_batch_size = 32

train_data_size = info.splits['train'].num_examples
steps_per_epoch = int(train_data_size / batch_size)
num_train_steps = steps_per_epoch * epochs
warmup_steps = int(0.1 * num_train_steps)
initial_learning_rate=2e-5

linear_decay = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=initial_learning_rate,
    end_learning_rate=0,
    decay_steps=num_train_steps)

warmup_schedule = tfm.optimization.lr_schedule.LinearWarmup(
    warmup_learning_rate = 0,
    after_warmup_lr_sched = linear_decay,
    warmup_steps = warmup_steps
)
optimizer = tf.keras.optimizers.experimental.Adam(
    learning_rate = warmup_schedule)

metrics = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy', dtype=tf.float32)]
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

bert_classifier.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=metrics)

bert_classifier.evaluate(glue_validation)
out_name = model_name
if out_name[:-1]=="/":
    out_name = out_name[:len(out_name)-1]
bert_classifier.save(model_name + ".h5")
bert_classifier.fit(
      glue_train,
      validation_data=(glue_validation),
      batch_size=batch_size,
      epochs=epochs)
bert_classifier.evaluate(glue_validation)
#tf.keras.utils.plot_model(bert_classifier, show_shapes=True, dpi=48)
#"""