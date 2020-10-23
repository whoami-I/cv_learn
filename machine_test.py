import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, datasets, Sequential, optimizers, metrics
import numpy as np

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
print(x_train.shape, y_train.shape)
# a = np.array([[1, 2, 3], [4, 5, 6]])
# b = np.array([True, False])
a = tf.random.normal([3, 2, 2])
b = tf.constant([False, True, False])
a = tf.data.Dataset.from_tensor_slices(tensors=(a, b))
print(list(a))
