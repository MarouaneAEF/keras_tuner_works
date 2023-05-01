import os
# shut INFO and WARNING messages up 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf 
import numpy as np 
import urllib.request

ROOT_URL = "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/"
BATCH_SIZE = 64
TRAIN_DATA = "FordA_TRAIN.tsv"
TEST_DATA = "FordA_TEST.tsv"


class generator(object):
    # class wise configuration : commun for all batches and generator should not take 
    # any argument 
    config = {
    "root_url" : ROOT_URL,
    "usage" : TRAIN_DATA,
    "batch_size": BATCH_SIZE,
    "shape" : 500
    
    }
    
    def __call__(self):
        url = self.config["root_url"] + self.config["usage"]
        with urllib.request.urlopen(url) as f:
            table = np.loadtxt(f, delimiter="\t")
            labels = table[:, 0]
            data = table[:, 1:]
            num_samples = data.shape[0]
            
            num_batches = num_samples // self.config["batch_size"]
            for i in range(0, num_batches):
                start = i * self.config["batch_size"]
                end = (i + 1) * self.config["batch_size"]
                data_batch = data[start:end]
                labels_batch = labels[start:end].reshape(-1)

                yield tf.expand_dims(data_batch, axis=-1) , labels_batch

def get_dataset(generator, usage):
    config = generator.config
    if usage == "training":
        generator.config["usage"] = TRAIN_DATA
    elif usage == "test":
        generator.config["usage"] = TEST_DATA
    else:
        raise ValueError(f"Invalid usage: {usage}")
    
    
    generator_inst = generator()
    
    dataset = tf.data.Dataset.from_generator(
        generator=generator_inst,
        output_types=(tf.float32, tf.int32),
        output_shapes=(
                (config["batch_size"], config["shape"], 1),
                config["batch_size"],
        )
    )

    dataset.map(normalize_data)
    
    assert dataset.element_spec[0].shape == (config["batch_size"], config["shape"], 1)
    assert dataset.element_spec[1].shape == (config["batch_size"],)

    return dataset.prefetch(tf.data.experimental.AUTOTUNE).cache()


def normalize_data(data, label):
    # perform normalization here
    data_norm = (data - tf.reduce_min(data, keepdims=True)) / (tf.reduce_max(data, keepdims=True) - tf.reduce_min(data, keepdims=True))
    data_norm = tf.reshape(data_norm, shape=tf.shape(data))
    label_hot = tf.where(label == -1, 0, label)
    label_hot = tf.reshape(label_hot, shape=tf.shape(label))
    
    return data_norm, label_hot