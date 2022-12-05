import os
import tempfile

import tensorflow as tf
import tensorflow_models as tfm
import tensorflow_datasets as tfds

import TransformerQuantization
import TransformerModel

import ConvertModel

os.environ["CUDA_VISIBLE_DEVICES"]="0"

partiton_config = {
    "intermediate_partitions":1,
    "fc_out_partitions":1,
    "embedding_partitions":1,
    "use_conv":False
}

model_name = "uncased_L-2_H-128_A-2"
model_dir = "models/" + model_name
epochs = 5

bert_classifier = tf.keras.models.load_model('bert_tiny.h5', custom_objects={
    'ScaledDotProduct':TransformerModel.ScaledDotProduct,
    'MultiHeadedAttention':TransformerModel.MultiHeadedAttention,
    'Dense':TransformerModel.Dense,
    'PartitionLayer':TransformerModel.PartitionLayer,
    'PartitionEmbedding':TransformerModel.PartitionEmbedding,
    'BertEmbedding':TransformerModel.BertEmbedding,
    'BERT':TransformerModel.BERT,
    'BertEncoder':TransformerModel.BertEncoder})

#strategy = tf.distribute.MirroredStrategy()#devices=["/gpu:0", "/gpu:1", "/gpu:2"])
strategy = tf.distribute.get_strategy()
BATCH_SIZE_PER_REPLICA = 16
batch_size = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

glue, info = tfds.load('glue/mrpc',
                       with_info=True,
                       batch_size=batch_size)

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

eval_batch_size = 32

train_data_size = info.splits['train'].num_examples
steps_per_epoch = int(train_data_size / batch_size)
num_train_steps = steps_per_epoch * epochs
warmup_steps = int(0.1 * num_train_steps)
initial_learning_rate=2e-5

linear_decay = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=initial_learning_rate,
    end_learning_rate=0,
    decay_steps=num_train_steps)

warmup_schedule = tfm.optimization.lr_schedule.LinearWarmup(
    warmup_learning_rate = 0,
    after_warmup_lr_sched = linear_decay,
    warmup_steps = warmup_steps
)
optimizer = tf.keras.optimizers.experimental.Adam(
    learning_rate = warmup_schedule)

metrics = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy', dtype=tf.float32)]
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

bert_classifier.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=metrics)
print("after")
bert_classifier.evaluate(glue_validation)
quant_aware_model = TransformerQuantization.QuantizeTransformer(bert_classifier)
tf.keras.backend.clear_session()

quant_aware_model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer='adam',
              metrics=tf.metrics.BinaryAccuracy(threshold=0.0))
quant_aware_model.summary()

quant_aware_model.fit(
    glue_train,
    validation_data=(glue_validation),
    batch_size=batch_size,
    epochs=epochs)

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

bert_inputs_processor = BertInputProcessor(tokenizer, packer)
glue_train = glue['train'].map(bert_inputs_processor).prefetch(1)
glue_validation = glue['validation'].map(bert_inputs_processor).prefetch(1)
glue_test = glue['test'].map(bert_inputs_processor).prefetch(1)

def representative_dataset():
  for data in glue_train.take(1):
    yield [tf.dtypes.cast(tf.dtypes.cast(data[0]["input_word_ids"], tf.float32), tf.float32),
    tf.dtypes.cast(tf.dtypes.cast(data[0]["input_mask"], tf.float32), tf.float32),
    tf.dtypes.cast(tf.dtypes.cast(data[0]["input_type_ids"], tf.float32), tf.float32)]

for input_idx in range(len(bert_classifier.input)):
    bert_classifier.input[input_idx].set_shape((1,) + bert_classifier.input[input_idx].shape[1:])


float_converter = tf.lite.TFLiteConverter.from_keras_model(bert_classifier)
float_tflite_model = float_converter.convert()
#open("tflite_models/mrpc_" + model_name + "_fp32.tflite", "wb").write(float_tflite_model)
#"""
converter = tf.lite.TFLiteConverter.from_keras_model(bert_classifier)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

debugger = tf.lite.experimental.QuantizationDebugger(
    converter=converter, debug_dataset=representative_dataset)
debugger.run()
RESULTS_FILE = 'debugger_results/uncased_L-2_H-128_A-2_debug.csv'
with open(RESULTS_FILE, 'w') as f:
  debugger.layer_statistics_dump(f)
quantized_tflite_model = debugger.get_nondebug_quantized_model()

_, quant_file = tempfile.mkstemp('.tflite')
_, float_file = tempfile.mkstemp('.tflite')
with open(float_file, 'wb') as f:
  f.write(float_tflite_model)

with open(quant_file, 'wb') as f:
  f.write(quantized_tflite_model)

print("Float model in Mb:", os.path.getsize(float_file) / float(2**20))
print("Quantized model in Mb:", os.path.getsize(quant_file) / float(2**20))

open("tflite_models/mrpc_" + model_name + "_int8.tflite", "wb").write(quantized_tflite_model)
open("tflite_models/mrpc_" + model_name + "_fp32.tflite", "wb").write(float_tflite_model)
#"""