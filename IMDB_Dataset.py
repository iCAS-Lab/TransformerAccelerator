
import matplotlib.pyplot as plt
import os
import re
import tempfile
import shutil
import string
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses
import tensorflow_model_optimization as tfmot
import numpy as np
import TransformerModel
import TransformerQuantization

print(tf.__version__)

"""## Sentiment analysis

This notebook trains a sentiment analysis model to classify movie reviews as *positive* or *negative*, based on the text of the review. This is an example of *binary*—or two-class—classification, an important and widely applicable kind of machine learning problem.

You'll use the [Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/) that contains the text of 50,000 movie reviews from the [Internet Movie Database](https://www.imdb.com/). These are split into 25,000 reviews for training and 25,000 reviews for testing. The training and testing sets are *balanced*, meaning they contain an equal number of positive and negative reviews.

### Download and explore the IMDB dataset

Let's download and extract the dataset, then explore the directory structure.
"""

def fetch_data(batch_size):
    seed = 42

    raw_train_ds = tf.keras.utils.text_dataset_from_directory(
        'aclImdb/train', 
        batch_size=batch_size, 
        validation_split=0.2, 
        subset='training', 
        seed=seed)

    raw_val_ds = tf.keras.utils.text_dataset_from_directory(
        'aclImdb/train', 
        batch_size=batch_size, 
        validation_split=0.2, 
        subset='validation', 
        seed=seed)

    raw_test_ds = tf.keras.utils.text_dataset_from_directory(
        'aclImdb/test', 
        batch_size=batch_size)

    def custom_standardization(input_data):
        lowercase = tf.strings.lower(input_data)
        stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
        return tf.strings.regex_replace(stripped_html,
                                        '[%s]' % re.escape(string.punctuation),
                                        '')

    max_features = 10000
    sequence_length = 250

    vectorize_layer = layers.TextVectorization(
        standardize=custom_standardization,
        max_tokens=max_features,
        output_mode='int',
        output_sequence_length=sequence_length)

    train_text = raw_train_ds.map(lambda x, y: x)
    vectorize_layer.adapt(train_text)


    def vectorize_text(text, label):
        text = tf.expand_dims(text, -1)
        return vectorize_layer(text), label

    train_ds = raw_train_ds.map(vectorize_text)
    val_ds = raw_val_ds.map(vectorize_text)
    test_ds = raw_test_ds.map(vectorize_text)

    """### Configure the dataset for performance

    These are two important methods you should use when loading data to make sure that I/O does not become blocking.

    `.cache()` keeps data in memory after it's loaded off disk. This will ensure the dataset does not become a bottleneck while training your model. If your dataset is too large to fit into memory, you can also use this method to create a performant on-disk cache, which is more efficient to read than many small files.

    `.prefetch()` overlaps data preprocessing and model execution while training. 

    You can learn more about both methods, as well as how to cache data to disk in the [data performance guide](https://www.tensorflow.org/guide/data_performance).
    """

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
    return train_ds, val_ds, test_ds