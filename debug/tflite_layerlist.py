import tensorflow as tf


# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="/home/user/Documents/TransformerAccelerator/detector.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# get details for each layer
all_layers_details = interpreter.get_tensor_details() 

for layer in all_layers_details:
    if layer["name"]=="class_net/class-predict/BiasAdd;class_net/class-predict/separable_conv2d_4;class_net/class-predict/separable_conv2d;class_net/class-predict/bias":
        print()
        print("*"*20)
        print(layer["name"], layer["shape"])
        print("*"*20)
        print()
    else:
        print(layer["name"], layer["shape"])