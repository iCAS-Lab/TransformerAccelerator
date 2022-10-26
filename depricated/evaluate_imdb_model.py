
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
    

val_x = np.loadtxt("data/IMDB_val_x")
val_y = np.loadtxt("data/IMDB_val_y")
print("Num samples", val_x.shape[0])

def evaluate_model(interpreter, x_test, y_test):
  input_index = interpreter.get_input_details()[0]["index"]
  output_index = interpreter.get_output_details()[0]["index"]

  # Run predictions on every image in the "test" dataset.
  acc = 0
  samples = 0
  for i, x_sample in enumerate(val_x):
    # Pre-processing: add batch dimension and convert to float32 to match with
    # the model's input data format.
    x_sample = np.reshape(x_sample, (1,x_sample.shape[0])).astype(np.float32)
    interpreter.set_tensor(input_index, x_sample)

    # Run inference.
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_index)
    output_data = float(1) if output_data[0][0] >= 0 else float(0)
    y_true = val_y[i]
    if output_data==y_true:
      acc+=1
    samples+=1

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