
import numpy as np
import tflite_runtime.interpreter as tflite
import numpy as np
import tempfile
import os
import string
import re
import time
import sys
if len(sys.argv) < 2:
  print("File not specified")
  print("Usage:",sys.argv[0], "<filename>.tflite")
  exit()
candidateA = sys.argv[1]

batch_size = 32
seed = 42
max_features = 4
sequence_length = 250
embedding_dim = 64
d_model = 512

test_samples = 400

val_x = np.loadtxt("data/mrpc_val_x.txt")
val_x = np.reshape(val_x, (val_x.shape[0], 3, 128))
val_y = np.loadtxt("data/mrpc_val_y.txt")
print("Num samples", val_x.shape[0])

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
    if i%10==0:
      print(i, "/", str(test_samples) + ":\tAccuracy:", acc / float(samples))

    # Post-processing: remove batch dimension and find the digit with highest
    # probability.
    #output = interpreter.tensor(output_index)
  print("Accuracy:", acc / float(samples))

currentBaseTime = time.time()
interpreter = tflite.Interpreter(candidateA)
interpreterReadTime1 = time.time() - currentBaseTime
currentBaseTime = time.time()

interpreter.allocate_tensors()
allocationTime1 = time.time() - currentBaseTime
currentBaseTime = time.time()

evaluate_model(interpreter, val_x, val_y)
evaluationTime1 = time.time() - currentBaseTime

print(candidateA.split("/")[-1])
print("Model Load Time is", interpreterReadTime1, "(secs)")
print("Allocation Time is", allocationTime1, "(secs)")
print("Inference Time is", evaluationTime1, "(secs)")
print("Total Time is", interpreterReadTime1 + allocationTime1 + evaluationTime1, "(secs)")