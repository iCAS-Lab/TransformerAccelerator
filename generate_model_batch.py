import os
import tempfile

import sys
if not len(sys.argv) > 5:
    print("usage:", sys.argv[0], "<model_name> <intr_partitions> <fc_out_part> <emb_part> <use_conv>")
    exit()


import tensorflow as tf
import tensorflow_models as tfm
import tensorflow_datasets as tfds

import ConvertModel

import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"]="-1"

model_name = sys.argv[1]
intr_part = int(sys.argv[2])
out_part = int(sys.argv[3])
emb_part = int(sys.argv[4])
use_conv = sys.argv[5]=="True"

partiton_config = {
    "intermediate_partitions":intr_part,
    "fc_out_partitions":out_part,
    "embedding_partitions":emb_part,
    "use_conv":use_conv
}

nicknames = {
    "intermediate_partitions":"intr",
    "fc_out_partitions":"out",
    "embedding_partitions":"emb",
    "use_conv":"conv"
}

model_info = pd.DataFrame()
model_info = model_info.append(partiton_config, ignore_index=True)
model_info["name"] = model_name

print("Generating model:", model_name, "with config:")

model_type = ""
for key in partiton_config:
    nickname = nicknames[key]
    value = str(partiton_config[key])
    model_type+=nickname + "-" + value + "_"
    print("\t" + key + ": " + str(value))
model_type = model_type[:-1]

out_dir = "models_for_paper/" + model_type
fp_dir = out_dir + "/fp32"
int8_dir = out_dir + "/int8"
edge_dir = out_dir + "/edgetpu"
if not os.path.exists(fp_dir):
    os.makedirs(fp_dir)
if not os.path.exists(int8_dir):
    os.makedirs(int8_dir)
if not os.path.exists(edge_dir):
    os.makedirs(edge_dir)

model_dir = "models/" + model_name

def fetchRawModel(batch_size=None):
    bert_encoder = ConvertModel.from_config(model_dir + "/bert_config.json", partition_config = partiton_config)
    return bert_encoder
import numpy as np
bert_classifier = fetchRawModel()
trainableParams = np.sum([np.prod(v.get_shape()) for v in bert_classifier.trainable_weights])
nonTrainableParams = np.sum([np.prod(v.get_shape()) for v in bert_classifier.non_trainable_weights])
totalParams = trainableParams + nonTrainableParams

model_info["trainable_params"] = trainableParams
model_info["non_trainableParams"] = nonTrainableParams
model_info["total_params"] = totalParams

bert_classifier.summary()

glue, info = tfds.load('glue/mrpc',
                       with_info=True,
                       batch_size=1)

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
steps_per_epoch = int(train_data_size / 1)
num_train_steps = steps_per_epoch * 1
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

tokenizer = tfm.nlp.layers.FastWordpieceBertTokenizer(
    vocab_file=os.path.join(model_dir, "vocab.txt"),
    lower_case=True)

max_seq_length = 128

packer = tfm.nlp.layers.BertPackInputs(
    seq_length=max_seq_length,
    special_tokens_dict = tokenizer.get_special_tokens_dict())

bert_inputs_processor = BertInputProcessor(tokenizer, packer)
glue_train = glue['train'].map(bert_inputs_processor).prefetch(1)
glue_validation = glue['validation'].map(bert_inputs_processor).prefetch(1)
glue_test = glue['test'].map(bert_inputs_processor).prefetch(1)

def representative_dataset():
  for data in glue_train.take(1):
    yield [tf.dtypes.cast(tf.dtypes.cast(data[0]["input_word_ids"], tf.float32), tf.float32),
    tf.dtypes.cast(tf.dtypes.cast(data[0]["input_mask"], tf.float32), tf.float32),
    tf.dtypes.cast(tf.dtypes.cast(data[0]["input_type_ids"], tf.float32), tf.float32)]

for input_idx in range(len(bert_classifier.input)):
    bert_classifier.input[input_idx].set_shape((1,) + bert_classifier.input[input_idx].shape[1:])


float_converter = tf.lite.TFLiteConverter.from_keras_model(bert_classifier)
float_tflite_model = float_converter.convert()

converter = tf.lite.TFLiteConverter.from_keras_model(bert_classifier)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

debugger = tf.lite.experimental.QuantizationDebugger(
    converter=converter, debug_dataset=representative_dataset)
debugger.run()
RESULTS_FILE = 'debugger_results/debugger_results.csv'
with open(RESULTS_FILE, 'w') as f:
  debugger.layer_statistics_dump(f)
quantized_tflite_model = debugger.get_nondebug_quantized_model()

_, quant_file = tempfile.mkstemp('.tflite')
_, float_file = tempfile.mkstemp('.tflite')
with open(float_file, 'wb') as f:
  f.write(float_tflite_model)

with open(quant_file, 'wb') as f:
  f.write(quantized_tflite_model)

print("Float model in Mb:", os.path.getsize(float_file) / float(2**20))
print("Quantized model in Mb:", os.path.getsize(quant_file) / float(2**20))

open("models_for_paper/" + model_type + "/fp32/" + model_name + "_fp32.tflite", "wb").write(float_tflite_model)
open("models_for_paper/" + model_type + "/int8/" + model_name + "_int8.tflite", "wb").write(quantized_tflite_model)
model_info.to_csv("models_for_paper/" + model_type + "/" + model_name + "_info.csv")