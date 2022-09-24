
from tflite import Model
import numpy as np
def get_methods(object, spacing=20):
  methodList = []
  for method_name in dir(object):
    try:
        if callable(getattr(object, method_name)):
            methodList.append(str(method_name))
    except Exception:
        methodList.append(str(method_name))
  processFunc = (lambda s: ' '.join(s.split())) or (lambda s: s)
  for method in methodList:
    try:
        print(str(method.ljust(spacing)) + ' ' +
              processFunc(str(getattr(object, method).__doc__)[0:90]))
    except Exception:
        print(method.ljust(spacing) + ' ' + ' getattr() failed')
model_path = '/home/user/Documents/TransformerAccelerator/tflite_models/mrpc_test_vehicle_int8.tflite'
buf = open(model_path, 'rb').read()
model = Model.GetRootAsModel(buf, 0)
subgraph = model.Subgraphs(0)
#print(get_methods(subgraph))
#print(get_methods(subgraph.Tensors(4)))
#print(subgraph.Tensors(4).Name())

TENSOR_TYPES = {
    "FLOAT32": 0,
    "FLOAT16": 1,
    "INT32": 2,
    "UINT8": 3,
    "INT64": 4,
    "STRING": 5,
    "BOOL": 6,
    "INT16": 7,
    "COMPLEX64": 8,
    "INT8": 9,
    "FLOAT64": 10,
    "COMPLEX128": 11,
    "UINT64": 12,
    "RESOURCE": 13,
    "VARIANT": 14,
    "UINT32": 15,
    "UINT16": 16
}

print(get_methods(subgraph))
for i in range(model.SubgraphsLength()):
    subgraph = model.Subgraphs(i)
    for j in range(subgraph.TensorsLength()):
        tensor = subgraph.Tensors(j)
        typeName = list(TENSOR_TYPES.keys())[list(TENSOR_TYPES.values()).index(tensor.Type())]
        tensorName = str(tensor.Name())
        tensorOperation = tensorName.split("/")[-1]
        variableName = '/'.join(tensorName.split("/")[:-1])
        quantizationParamas = tensor.Quantization()
        buffer = tensor.Buffer()
        raw = np.array(model.Buffers(buffer).DataAsNumpy())
        op = subgraph.Operators(j)
        scale = quantizationParamas.ScaleAsNumpy()
        zero_point = quantizationParamas.ZeroPointAsNumpy()
        #print(tensorOperation, tensor.IsVariable(), tensor.ShapeAsNumpy(), typeName, sep="\t")
        #print(tensorOperation, ">",end="")
        if "layer_0" in tensorName and not "batchnorm" in tensorName:
            print(tensorName, typeName, scale[0], zero_point[0], raw.shape)

#print(get_methods(subgraph))
#print(get_methods(model.Subgraphs(0).Tensors(0)))

#print(get_methods(model))