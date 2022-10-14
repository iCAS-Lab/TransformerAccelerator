
import numpy as np
import tflite_runtime.interpreter as tflite
import numpy as np
import tempfile
import os
import string
import re
import time
import sys
import lightweight_dataframes as dataframes

test_samples = 400

if len(sys.argv) < 2:
  print("File not specified")
  print("Usage:",sys.argv[0], "<model_directory>")
  exit()

model_dir = sys.argv[1]

out_file = os.path.join(model_dir, "benchmark.csv")
if len(sys.argv)==3:
  out_file = sys.argv[2]

files = os.listdir(model_dir)

val_x = np.loadtxt("data/mrpc_val_x.txt")
val_x = np.reshape(val_x, (val_x.shape[0], 3, 128))
val_y = np.loadtxt("data/mrpc_val_y.txt")
val_x = val_x[:test_samples]
val_y = val_y[:test_samples]
print("Num samples", test_samples)

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
  return acc / float(samples)
df = dataframes.createDataFrame(columnNames=["nickname", "accuracy", "avg inference time (ms)", "size (mb)",
              "alloc_time (ms)", "total_inference_time (s)", "load_time (ms)", "num_samples", "file_name"])
min_same = 999999999
filename = files[0]
for sub_start in range(len(filename)):
  for filename2 in files[1:]:
    if not ".tflite" in filename2 or filename==filename2:
      continue
    curr_same = 0
    for i, char in enumerate(filename2):
      if char==filename[i]:
        curr_same+=1
      else:
        break
    min_same = min(curr_same, min_same)

for filename in files:
  if not ".tflite" in filename:
    continue
  nickname = filename[min_same:-len(".tflite")]
  print("Evaluating:", filename)
  model_path = os.path.join(model_dir, filename)
  model_size = os.path.getsize(model_path) / float(2**20)

  currentBaseTime = time.time()
  if "edgetpu" in model_path:
    interpreter = tflite.Interpreter(model_path, experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
  else:
    interpreter = tflite.Interpreter(model_path, num_threads=2)
  interpreterReadTime1 = (time.time() - currentBaseTime)*1000 # ms
  currentBaseTime = time.time()

  interpreter.allocate_tensors()
  allocationTime1 = (time.time() - currentBaseTime)*1000 # ms
  currentBaseTime = time.time()

  acc = evaluate_model(interpreter, val_x, val_y)
  evaluationTime1 = time.time() - currentBaseTime

  avg_infer_time = (evaluationTime1 / test_samples)*1000

  print(model_path.split("/")[-1])
  print("Model Load Time is", interpreterReadTime1, "(ms)")
  print("Allocation Time is", allocationTime1, "(ms)")
  print("Inference Time is", evaluationTime1, "(secs)")
  print("Total Time is", interpreterReadTime1/1000 + allocationTime1/1000 + evaluationTime1, "(secs)")
  print("Avg Time for one inference:", avg_infer_time, "(ms)")
  df = dataframes.append_row(df, {"nickname":nickname, "accuracy":acc, "avg inference time (ms)":avg_infer_time, "size (mb)":model_size,
              "alloc_time (ms)":allocationTime1, "total_inference_time (s)":evaluationTime1, "load_time (ms)":interpreterReadTime1, "num_samples":test_samples, "file_name":filename})
  dataframes.to_csv(df, out_file)

dataframes.print_df(df)
dataframes.to_csv(df, out_file)