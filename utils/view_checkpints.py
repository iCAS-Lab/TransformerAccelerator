from tensorflow.python.training import py_checkpoint_reader

# Need to say "model.ckpt" instead of "model.ckpt.index" for tf v2
file_name = "/home/brendan/bert-tiny/uncased_L-2_H-128_A-2/bert_model.ckpt"
reader = py_checkpoint_reader.NewCheckpointReader(file_name)

state_dict = {
    v: reader.get_tensor(v) for v in reader.get_variable_to_shape_map()
}

for i in state_dict.keys():
    try:
        print(i, state_dict[i].shape)
    except:
        print("<<<<", i)