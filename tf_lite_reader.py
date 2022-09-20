import tensorflow as tf
tflite_interpreter = tf.lite.Interpreter(model_path='/home/user/Documents/TransformerAccelerator/tflite_models/imdb_test_vehicle_int8.tflite')
tflite_interpreter.allocate_tensors()

tensor_details = tflite_interpreter.get_tensor_details()
for dict in tensor_details:
    i = dict['index']
    tensor_name = dict['name']
    scales = dict['quantization_parameters']['scales']
    zero_points = dict['quantization_parameters']['zero_points']
    tensor = tflite_interpreter.tensor(i)()

    print(i, tensor_name, scales, zero_points, tensor.shape)