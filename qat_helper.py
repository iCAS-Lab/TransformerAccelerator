import sys
if len(sys.argv) < 2:
    print("Usage: ", sys.argv[0], " <out_name>")
    exit()

OUT_NAME = sys.argv[1]

import os
import tempfile

import tensorflow as tf
import tensorflow_models as tfm
import tensorflow_datasets as tfds

import TransformerModel
import QuantizedTransformer

import ConvertModel
import math
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"]="3"

model_name = "uncased_L-8_H-512_A-8"
model_dir = "models/" + model_name
epochs = 10

partiton_config = {
    "intermediate_partitions":1,
    "fc_out_partitions":1,
    "embedding_partitions":1,
    "use_conv":False
}

bert_classifier = tf.keras.models.load_model(OUT_NAME + '.h5', custom_objects={
    'ScaledDotProduct':TransformerModel.ScaledDotProduct,
    'MultiHeadedAttention':TransformerModel.MultiHeadedAttention,
    'ConfigurableDense':TransformerModel.ConfigurableDense,
    'PartitionLayer':TransformerModel.PartitionLayer,
    'PartitionEmbedding':TransformerModel.PartitionEmbedding,
    'BertEmbedding':TransformerModel.BertEmbedding,
    'BERT':TransformerModel.BERT,
    'BertEncoder':TransformerModel.BertEncoder})
#strategy = tf.distribute.MirroredStrategy()#devices=["/gpu:0", "/gpu:1", "/gpu:2"])
strategy = tf.distribute.get_strategy()
BATCH_SIZE_PER_REPLICA = 8
batch_size = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

glue, info = tfds.load('glue/mrpc',
                       with_info=True,
                       batch_size=batch_size)

tokenizer = tfm.nlp.layers.FastWordpieceBertTokenizer(
    vocab_file=os.path.join(model_dir, "vocab.txt"),
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
    learning_rate = initial_learning_rate) #change

metrics = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy', dtype=tf.float32)]
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

bert_classifier.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=metrics)
print("loaded model accuracy:")
bert_classifier.evaluate(glue_validation)

class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        quant_aware_model.get_layer('transformer').on_epoch()
        
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
    learning_rate = 2e-5)

metrics = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy', dtype=tf.float32)]
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

optimizer = tf.keras.optimizers.experimental.Adam(
    learning_rate =warmup_schedule)

quant_aware_model = ConvertModel.clone_from_archtype(bert_classifier, model_dir + "/bert_config.json", archtype=QuantizedTransformer)
quant_aware_model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics)
print("loaded_q_aware")
quant_aware_model.evaluate(glue_validation)
history = quant_aware_model.fit(
    glue_train,
    validation_data=(glue_validation),
    batch_size=batch_size,
    epochs=epochs)

#quant_aware_model.save("q_aware_" + OUT_NAME + ".h5", include_optimizer=False)
quant_aware_model = QuantizedTransformer.DequantizedModel(quant_aware_model, model_dir + "/bert_config.json")
      
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

quant_aware_model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=metrics)
quant_aware_model.evaluate(glue_validation)
quant_aware_model.save("range_based_" + OUT_NAME + ".h5", include_optimizer=False)