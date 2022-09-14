
import tensorflow_datasets as tfds
import tensorflow_models as tfm
import tensorflow as tf
import os
import numpy as np
model_dir = "/home/brendan/TransformerAccelerator/models/uncased_L-12_H-768_A-12"
glue, info = tfds.load('glue/mrpc',
                       with_info=True,
                       batch_size=1)

tokenizer = tfm.nlp.layers.FastWordpieceBertTokenizer(
    vocab_file=os.path.join(model_dir, "vocab.txt"),
    lower_case=True)

max_seq_length = 128

packer = tfm.nlp.layers.BertPackInputs(
    seq_length=max_seq_length,
    special_tokens_dict = tokenizer.get_special_tokens_dict())

class BertInputProcessor(tf.keras.layers.Layer):
    def __init__(self, tokenizer, packer):
        super().__init__()
        self.tokenizer = tokenizer
        self.packer = packer

    def call(self, inputs):
        tok1 = self.tokenizer(inputs['sentence1'])
        tok2 = self.tokenizer(inputs['sentence2'])

        packed = self.packer([tok1, tok2])

        if 'label' in inputs:
            return packed, inputs['label']
        else:
            return packed

bert_inputs_processor = BertInputProcessor(tokenizer, packer)
glue_train = glue['train'].map(bert_inputs_processor).prefetch(1)
glue_validation = glue['validation'].map(bert_inputs_processor).prefetch(1)
glue_test = glue['test'].map(bert_inputs_processor).prefetch(1)
def representative_dataset(num_samples):
    x_arr = np.zeros((num_samples,3,128))
    y_arr = np.zeros((num_samples,))
    for i,data in enumerate(glue_validation.take(num_samples)):
        x_arr[i][0] = tf.dtypes.cast(data[0]["input_word_ids"], tf.float32)
        x_arr[i][1] = tf.dtypes.cast(data[0]["input_mask"], tf.float32)
        x_arr[i][2] = tf.dtypes.cast(data[0]["input_type_ids"], tf.float32)
        
        y_arr[i] = data[1]
    return x_arr, y_arr
num_samples = 500
x_arr, y_arr = representative_dataset(num_samples)
np.savetxt("data/mrpc_val_x.txt", np.reshape(x_arr, (num_samples, 3*128)))
np.savetxt("data/mrpc_val_y.txt", y_arr)