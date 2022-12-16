import os
import tempfile

import tensorflow as tf
import tensorflow_models as tfm
import tensorflow_datasets as tfds

import TransformerModel

import ConvertModel
import numpy as np

model_name = "mnist_classifier"
tf.keras.models.load_model(model_name + '.h5')


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
#"""
ptq_converter = tf.lite.TFLiteConverter.from_keras_model(bert_classifier)
ptq_converter.optimizations = [tf.lite.Optimize.DEFAULT]
ptq_converter.representative_dataset = representative_dataset
ptq_converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

converter = tf.lite.TFLiteConverter.from_keras_model(quant_aware_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

debugger = tf.lite.experimental.QuantizationDebugger(
    converter=converter, debug_dataset=representative_dataset)
debugger.run()
RESULTS_FILE = 'debugger_results/QAT' + model_name + '_kernel_axis-2.csv'
with open(RESULTS_FILE, 'w') as f:
  debugger.layer_statistics_dump(f)
quantized_tflite_model = debugger.get_nondebug_quantized_model()

debugger = tf.lite.experimental.QuantizationDebugger(
    converter=ptq_converter, debug_dataset=representative_dataset)
debugger.run()
RESULTS_FILE = 'debugger_results/PTQ' + model_name + '_kernel_axis-2.csv'
with open(RESULTS_FILE, 'w') as f:
  debugger.layer_statistics_dump(f)
ptq_quantized_tflite_model = debugger.get_nondebug_quantized_model()

_, quant_file = tempfile.mkstemp('.tflite')
_, float_file = tempfile.mkstemp('.tflite')
with open(float_file, 'wb') as f:
  f.write(float_tflite_model)

with open(quant_file, 'wb') as f:
  f.write(quantized_tflite_model)

print("Float model in Mb:", os.path.getsize(float_file) / float(2**20))
print("Quantized model in Mb:", os.path.getsize(quant_file) / float(2**20))

open("tflite_models/" + model_name + "_int8.tflite", "wb").write(ptq_quantized_tflite_model)
open("tflite_models/" + model_name + "_fp32.tflite", "wb").write(float_tflite_model)

print("Calculating model accuracy for validation subset")

def evaluate_model(interpreter, x_test, y_test):
  output_index = interpreter.get_output_details()[0]["index"]

  word_ids_index = interpreter.get_input_details()[0]["index"]
  input_type_ids_index = interpreter.get_input_details()[1]["index"]
  input_mask_index = interpreter.get_input_details()[2]["index"]

  acc = 0
  samples = 0
  for i, x_sample in enumerate(val_x[:test_samples]):
    word_ids = np.reshape(x_sample[0], (1,x_sample.shape[1])).astype(np.float32)
    mask = np.reshape(x_sample[1], (1,x_sample.shape[1])).astype(np.float32)
    type_ids = np.reshape(x_sample[2], (1,x_sample.shape[1])).astype(np.float32)


    interpreter.set_tensor(word_ids_index, word_ids)
    interpreter.set_tensor(input_mask_index, mask)
    interpreter.set_tensor(input_type_ids_index, type_ids)

    # Run inference.
    interpreter.invoke()

    out_1 = interpreter.get_tensor(output_index)
    out_class = int(np.argmax(out_1))
    if out_class == int(val_y[i]):
      acc+=1
    samples+=1
  return acc / float(samples)

interpreter = tf.lite.Interpreter(model_name + "_int8.tflite")
interpreter.allocate_tensors()
int8acc = evaluate_model(interpreter, val_x, val_y)

interpreter = tf.lite.Interpreter(model_name + "_fp32.tflite")
interpreter.allocate_tensors()
fp32acc = evaluate_model(interpreter, val_x, val_y)

print("8-bit INT Acc:", int8acc)
print("32-bit FP Acc:", fp32acc)