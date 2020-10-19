import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, datasets, Sequential, optimizers, metrics
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


# 数据预处理
def train_preprocess(x_train, y_train):
    # x_train = tf.cast(x=x_train, dtype=tf.float32) / 255.
    x_train = tf.cast(tf.reshape(x_train, [-1]), dtype=tf.float32) / 255.
    y_train = tf.cast(x=y_train, dtype=tf.int32)
    y_train = tf.one_hot(indices=y_train, depth=10)

    return x_train, y_train


db_val = tf.data.Dataset.from_tensor_slices((x_test,y_test))
db_val = db_val.map(train_preprocess).batch(128)

model = tf.keras.models.load_model('number_net2')
xtest = tf.reshape(x_test, shape=[-1, 28 * 28])
loss, acc = model.evaluate(db_val)
print(loss, acc)
