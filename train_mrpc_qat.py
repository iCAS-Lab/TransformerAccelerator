import os
import tempfile

import tensorflow as tf
import tensorflow_models as tfm
import tensorflow_datasets as tfds

import TransformerModel
import QuantizedTransformer

import ConvertModel
import numpy as np
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"]="0"

partiton_config = {
    "intermediate_partitions":1,
    "fc_out_partitions":1,
    "embedding_partitions":1,
    "use_conv":False
}

model_name = "uncased_L-8_H-512_A-8"
model_dir = "models/" + model_name
epochs = 6

bert_classifier = tf.keras.models.load_model(model_name + '.h5', custom_objects={
    'ScaledDotProduct':TransformerModel.ScaledDotProduct,
    'MultiHeadedAttention':TransformerModel.MultiHeadedAttention,
    'ConfigurableDense':TransformerModel.ConfigurableDense,
    'PartitionLayer':TransformerModel.PartitionLayer,
    'ConfigurableDense':TransformerModel.ConfigurableDense,
    'PartitionEmbedding':TransformerModel.PartitionEmbedding,
    'BertEmbedding':TransformerModel.BertEmbedding,
    'BERT':TransformerModel.BERT,
    'BertEncoder':TransformerModel.BertEncoder})
#strategy = tf.distribute.MirroredStrategy()#devices=["/gpu:0", "/gpu:1", "/gpu:2"])
strategy = tf.distribute.get_strategy()
BATCH_SIZE_PER_REPLICA = 16
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
    learning_rate = warmup_schedule)

metrics = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy', dtype=tf.float32)]
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

bert_classifier.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=metrics)
print("loaded model accuracy:")
bert_classifier.evaluate(glue_validation)
#quant_aware_model = QuantizedTransformer.QuantizationAwareModel(bert_classifier)
quant_aware_model = ConvertModel.clone_from_archtype(bert_classifier, model_dir + "/bert_config.json", archtype=QuantizedTransformer)

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
"""
metrics = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy', dtype=tf.float32)]
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
import math
import pandas as pd
out_list = []
for i in range(epochs):
    #epoch = min(20, i)
    epoch = i
    alpha = 1-(min(5,epoch)/5)
    #alpha = 1.37*(1/(1+math.exp(0.25*epoch-1)))
    #if alpha <= 0.3:
    #    alpha = 0
    beta = 1-alpha
    print("Epoch:", i, "\tAlpha:", alpha, "Beta:", beta)
    quant_aware_model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics)
    history = quant_aware_model.fit(
        glue_train,
        validation_data=(glue_validation),
        batch_size=batch_size,
        epochs=1,
        callbacks=[CustomCallback()])
    hist_dict = history.history
    for key in hist_dict.keys():
        hist_dict[key] = hist_dict[key][0]
    out_list.append(hist_dict | {"alpha":alpha, "beta":beta})
    out_csv = pd.DataFrame(out_list)
    out_csv.to_csv(model_name + "_linear_mean5_results.csv")

optimizer = tf.keras.optimizers.experimental.Adam(
    learning_rate =warmup_schedule)
quant_aware_model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics)
history = quant_aware_model.fit(
    glue_train,
    validation_data=(glue_validation),
    batch_size=batch_size,
    epochs=10)

quant_aware_model.evaluate(glue_validation)
quant_aware_model.save("q_aware_" + model_name + ".h5")
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
quant_aware_model.save("range_based_" + model_name + ".h5")
"""
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

bert_inputs_processor = BertInputProcessor(tokenizer, packer)
glue_train = glue['train'].map(bert_inputs_processor).prefetch(1)
glue_validation = glue['validation'].map(bert_inputs_processor).prefetch(1)
glue_test = glue['test'].map(bert_inputs_processor).prefetch(1)

def representative_dataset():
  for data in glue_train.take(400):
    yield [tf.dtypes.cast(tf.dtypes.cast(data[0]["input_word_ids"], tf.float32), tf.float32),
    tf.dtypes.cast(tf.dtypes.cast(data[0]["input_mask"], tf.float32), tf.float32),
    tf.dtypes.cast(tf.dtypes.cast(data[0]["input_type_ids"], tf.float32), tf.float32)]

for input_idx in range(len(bert_classifier.input)):
    bert_classifier.input[input_idx].set_shape((1,) + bert_classifier.input[input_idx].shape[1:])

float_converter = tf.lite.TFLiteConverter.from_keras_model(bert_classifier)
float_tflite_model = float_converter.convert()
#open("tflite_models/mrpc_" + model_name + "_fp32.tflite", "wb").write(float_tflite_model)
"""
ptq_converter = tf.lite.TFLiteConverter.from_keras_model(bert_classifier)
ptq_converter.optimizations = [tf.lite.Optimize.DEFAULT]
ptq_converter.representative_dataset = representative_dataset
ptq_converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
"""
converter = tf.lite.TFLiteConverter.from_keras_model(bert_classifier)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

debugger = tf.lite.experimental.QuantizationDebugger(
    converter=converter, debug_dataset=representative_dataset)
debugger.run()
RESULTS_FILE = 'debugger_results/QAT' + model_name + '_mixed_precision.csv'
with open(RESULTS_FILE, 'w') as f:
  debugger.layer_statistics_dump(f)
layer_stats = pd.read_csv(RESULTS_FILE)
layer_stats['range'] = 255.0 * layer_stats['scale']
layer_stats['rmse/scale'] = layer_stats.apply(
    lambda row: np.sqrt(row['mean_squared_error']) / row['scale'], axis=1)
#suspected_layers = list(
    #layer_stats[layer_stats['rmse/scale'] > 0.29]['tensor_name'])

suspected_layers = list(layer_stats['tensor_name'][:-1])
#quantized_tflite_model = debugger.get_nondebug_quantized_model()
debug_options = tf.lite.experimental.QuantizationDebugOptions(
    denylisted_nodes=suspected_layers)
debugger = tf.lite.experimental.QuantizationDebugger(
    converter=converter,
    debug_dataset=representative_dataset,
    debug_options=debug_options)
quantized_tflite_model = debugger.get_nondebug_quantized_model()
"""
debugger = tf.lite.experimental.QuantizationDebugger(
    converter=ptq_converter, debug_dataset=representative_dataset)
debugger.run()
RESULTS_FILE = 'debugger_results/PTQ' + model_name + '_mixed_precision.csv'
with open(RESULTS_FILE, 'w') as f:
  debugger.layer_statistics_dump(f)
ptq_quantized_tflite_model = debugger.get_nondebug_quantized_model()
"""
_, quant_file = tempfile.mkstemp('.tflite')
_, float_file = tempfile.mkstemp('.tflite')
with open(float_file, 'wb') as f:
  f.write(float_tflite_model)

with open(quant_file, 'wb') as f:
  f.write(quantized_tflite_model)

print("Float model in Mb:", os.path.getsize(float_file) / float(2**20))
print("Quantized model in Mb:", os.path.getsize(quant_file) / float(2**20))

open("tflite_models/mrpc_" + model_name + "_QAT_int8.tflite", "wb").write(quantized_tflite_model)
#open("tflite_models/mrpc_" + model_name + "_PTQ_int8.tflite", "wb").write(ptq_quantized_tflite_model)
open("tflite_models/mrpc_" + model_name + "_fp32.tflite", "wb").write(float_tflite_model)


test_samples = 400

val_x = np.loadtxt("data/mrpc_val_x.txt")
val_x = np.reshape(val_x, (val_x.shape[0], 3, 128))
val_y = np.loadtxt("data/mrpc_val_y.txt")
print("Calculating model accuracy for validation subset")

def evaluate_model(interpreter, x_test, y_test):
  output_index = interpreter.get_output_details()[0]["index"]

  word_ids_index = interpreter.get_input_details()[0]["index"]
  input_type_ids_index = interpreter.get_input_details()[1]["index"]
  input_mask_index = interpreter.get_input_details()[2]["index"]

  #print(interpreter.get_input_details()[0])
  #print(interpreter.get_input_details()[1])
  #print(interpreter.get_input_details()[2])
  # Run predictions on every image in the "test" dataset.
  acc = 0
  samples = 0
  for i, x_sample in enumerate(val_x[:test_samples]):
    # Pre-processing: add batch dimension and convert to float32 to match with
    # the model's input data format.
    word_ids = np.reshape(x_sample[0], (1,x_sample.shape[1])).astype(np.float32)
    mask = np.reshape(x_sample[1], (1,x_sample.shape[1])).astype(np.float32)
    type_ids = np.reshape(x_sample[2], (1,x_sample.shape[1])).astype(np.float32)

    #print("WORDS", word_ids)
    #print("MASK", mask)
    #print("TYPE_IDS",type_ids)

    interpreter.set_tensor(word_ids_index, word_ids)
    interpreter.set_tensor(input_mask_index, mask)
    interpreter.set_tensor(input_type_ids_index, type_ids)

    # Run inference.
    interpreter.invoke()

    out_1 = interpreter.get_tensor(output_index)
    #print(out_1)
    out_class = int(np.argmax(out_1))
    if out_class == int(val_y[i]):
      acc+=1
    samples+=1
    # Post-processing: remove batch dimension and find the digit with highest
    # probability.
    #output = interpreter.tensor(output_index)
  return acc / float(samples)

interpreter = tf.lite.Interpreter("tflite_models/mrpc_" + model_name + "_QAT_int8.tflite")
interpreter.allocate_tensors()
QATint8acc = evaluate_model(interpreter, val_x, val_y)

#interpreter = tf.lite.Interpreter("tflite_models/mrpc_" + model_name + "_PTQ_int8.tflite")
#interpreter.allocate_tensors()
#PTQint8acc = evaluate_model(interpreter, val_x, val_y)

interpreter = tf.lite.Interpreter("tflite_models/mrpc_" + model_name + "_fp32.tflite")
interpreter.allocate_tensors()
fp32acc = evaluate_model(interpreter, val_x, val_y)

print("8-bit QAT Acc:", QATint8acc)
#print("8-bit PTQ Acc:", PTQint8acc)
print("32-bit FP Acc:", fp32acc)
#"""