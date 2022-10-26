
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

test_samples = 100
n_threads = 1

if len(sys.argv) < 2:
  print("File not specified")
  print("Usage:",sys.argv[0], "<model_directory>")
  exit()

results_name = "benchmark.csv"
if n_threads > 1:
  results_name = "thread-" + str(n_threads) + results_name

model_dir = sys.argv[1]
model_type = model_dir
print("Model type:", model_type)
if len(model_type.split("/")) > 1:
  if len(model_type.split("/")[-1]) == 0:
    model_type = model_type.split("/")[-2]
  else:
    model_type = model_type.split("/")[-1]


print("Model type:", model_type)
out_file = os.path.join(model_dir, results_name)

files = os.listdir(model_dir)


val_x = np.loadtxt("data/mrpc_val_x.txt")
val_x = np.reshape(val_x, (val_x.shape[0], 3, 128))
val_y = np.loadtxt("data/mrpc_val_y.txt")
print("Num samples", test_samples)
output_index = None

word_ids_index = None
input_type_ids_index = None
input_mask_index = None

def evaluate_model(interpreter, word_ids, mask, type_ids):
  interpreter.set_tensor(word_ids_index, word_ids)
  interpreter.set_tensor(input_mask_index, mask)
  interpreter.set_tensor(input_type_ids_index, type_ids)

  interpreter.invoke()
  
  out_1 = interpreter.get_tensor(output_index)
  return 0
df = dataframes.createDataFrame(columnNames=["nickname", "var", "accuracy", "avg inference time (ms)", "size (mb)",
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
    interpreter = tflite.Interpreter(model_path, num_threads=n_threads, experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
  else:
    interpreter = tflite.Interpreter(model_path, num_threads=n_threads)
  
  output_index = interpreter.get_output_details()[0]["index"]

  word_ids_index = interpreter.get_input_details()[0]["index"]
  input_type_ids_index = interpreter.get_input_details()[1]["index"]
  input_mask_index = interpreter.get_input_details()[2]["index"]

  interpreterReadTime1 = (time.time() - currentBaseTime)*1000 # ms
  currentBaseTime = time.time()

  interpreter.allocate_tensors()
  allocationTime1 = (time.time() - currentBaseTime)*1000 # ms
  infer_times = []
  x_sample = val_x[0]
  word_ids = np.reshape(x_sample[0], (1,x_sample.shape[1])).astype(np.float32)
  mask = np.reshape(x_sample[1], (1,x_sample.shape[1])).astype(np.float32)
  type_ids = np.reshape(x_sample[2], (1,x_sample.shape[1])).astype(np.float32)
  acc = evaluate_model(interpreter, word_ids, mask, type_ids)
  for i in range(test_samples):
    x_sample = val_x[i]
    word_ids = np.reshape(x_sample[0], (1,x_sample.shape[1])).astype(np.float32)
    mask = np.reshape(x_sample[1], (1,x_sample.shape[1])).astype(np.float32)
    type_ids = np.reshape(x_sample[2], (1,x_sample.shape[1])).astype(np.float32)
    currentBaseTime = time.time()
    acc = evaluate_model(interpreter, word_ids, mask, type_ids)
    evaluationTime = time.time() - currentBaseTime
    infer_times.append(evaluationTime)

  #infer_times = infer_times[1:]
  avg_infer_time = (sum(infer_times) / len(infer_times))*1000
  var_infer_time = np.var(infer_times)*1000
  infer_range = (max(infer_times) - min(infer_times))*1000
  infer_times = [x*1000 for x in infer_times]
  print("="*50)
  print(model_path.split("/")[-1])
  print("Avg Time for one inference:", avg_infer_time, "(ms)")
  print("Inference variance:", var_infer_time, "(ms)")
  print("Inference range:", infer_range, "(ms)")
  print("="*50)
  df = dataframes.append_row(df, {"nickname":nickname, "var":var_infer_time, "accuracy":acc, "avg inference time (ms)":avg_infer_time, "size (mb)":model_size,
              "alloc_time (ms)":allocationTime1, "total_inference_time (s)":sum(infer_times), "load_time (ms)":interpreterReadTime1, "num_samples":test_samples, "file_name":filename})
  dataframes.to_csv(df, out_file)

dataframes.print_df(df)
dataframes.to_csv(df, out_file)