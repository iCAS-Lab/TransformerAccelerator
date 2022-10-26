import time
start_time = time.time()
import tflite_runtime.interpreter as tflite
import numpy as np
import os
import sys
import lightweight_dataframes as dataframes
n_threads = 1
inference_time = 300

if len(sys.argv) < 2:
  print("File not specified")
  print("Usage:",sys.argv[0], "<model.tflite>")
  exit()
print("Benchmarking power for", sys.argv[1])

filename = sys.argv[1]
model_name = filename.split("/")[-1][:-7]
######################
### MODEL LOADING ####
######################
currentBaseTime = time.time()

if "edgetpu" in filename:
    interpreter = tflite.Interpreter(filename, num_threads=n_threads, experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
else:
    interpreter = tflite.Interpreter(filename, num_threads=n_threads)

modelLoadTime = (time.time() - currentBaseTime)
######################
## MODEL ALLOCATION ##
######################
currentBaseTime = time.time()

df = dataframes.createDataFrame(columnNames=["model_name", "file_name", "num_inferences", "total_time (s)", "load_time (s)", "alloc_time (s)", "inference_time (s)", "cpu_threads"])
num_infer = 0
avg_time = 0

interpreter.allocate_tensors()

output_index = interpreter.get_output_details()[0]["index"]

word_ids_index = interpreter.get_input_details()[0]["index"]
input_type_ids_index = interpreter.get_input_details()[1]["index"]
input_mask_index = interpreter.get_input_details()[2]["index"]

allocationTime = (time.time() - currentBaseTime)
######################
## MODEL INFERENCE ###
######################
inference_start_time = time.time()

def infer(input_ids, type_ids, mask_ids):
    interpreter.set_tensor(word_ids_index, input_ids)
    interpreter.set_tensor(input_mask_index, mask_ids)
    interpreter.set_tensor(input_type_ids_index, type_ids)

    interpreter.invoke()
    
    out = interpreter.get_tensor(output_index)
    return [out]

while time.time() - inference_start_time < inference_time:
    input_ids = np.random.randint(10000, size=(1,128)).astype(np.float32)
    token_type_ids = np.random.randint(2, size=(1,128)).astype(np.float32)
    input_mask_ids = np.random.randint(1, size=(1,128)).astype(np.float32)
    outs = infer(input_ids, token_type_ids, input_mask_ids)
    num_infer+=1

inferenceTime = time.time() - inference_start_time
totalTime = time.time() - start_time
print(filename)
print("-----------------")
print("POWER     PROFILE")
print("")
print("Number of samples analyzed:\t", num_infer)
print("Model Load Time:           \t", modelLoadTime, "(s)")
print("Model Alloc Time:          \t", allocationTime, "(s)")
print("Model Inerence Time:       \t", inferenceTime, "(s)")
print("Total Program Runtime:     \t", totalTime, "(s)")


df = dataframes.append_row(df, {"model_name":model_name,"file_name":filename, "num_inferences":num_infer, "total_time (s)":totalTime, "load_time (s)":modelLoadTime, "alloc_time (s)":allocationTime, "inference_time (s)":inferenceTime, "cpu_threads":n_threads})
dataframes.to_csv(df, model_name + "_power_timeline.csv")
