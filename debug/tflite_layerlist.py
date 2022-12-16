import tensorflow as tf


# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="/home/user/Documents/TransformerAccelerator/tflite_models/mrpc_uncased_L-8_H-512_A-8qat_sdp-layernorm_fp32.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# get details for each layer
all_layers_details = interpreter.get_tensor_details() 

for layer in all_layers_details:
    if "partition0" in layer["name"] and not "norm" in layer["name"]:
        print(layer)
        #print(layer["name"], layer["shape"], layer["quantization"])