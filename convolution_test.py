import tensorflow as tf
import numpy as np
import timeit
from tensorflow.keras import datasets
import tensorflow.keras

# input = tf.constant([[1, 2, 3, 4, 5],
#                      [1, 2, 3, 4, 5],
#                      [1, 2, 3, 4, 5],
#                      [1, 2, 3, 4, 5],
#                      [1, 2, 3, 4, 5]])
# input = tf.expand_dims(input, axis=2)
# input = tf.expand_dims(input, axis=0)
# w = [[-1, -1, -1], [-1, 5, -1], [-1, -1, -1]]
# w = tf.expand_dims(tf.expand_dims(w, axis=2), axis=3)
# output = tf.nn.conv2d(input, w, 1, padding=[[0, 0], [0, 0], [0, 0], [0, 0]])
# print(input.shape)
# print(w.shape)
# print(output)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


# print("x_test", x_test.shape, "y_test", y_test.shape)


# 数据预处理
def train_preprocess(x_train, y_train):
    # x_train = tf.cast(x=x_train, dtype=tf.float32) / 255.
    # x_train = tf.cast(tf.reshape(x_train, [-1]), dtype=tf.float32)
    x_train = tf.cast(x_train, dtype=tf.float32)
    x_train = tf.expand_dims(x_train, axis=2)
    # x_train = tf.expand_dims(tf.expand_dims(x_train, axis=2), axis=0)
    print('----------------------', x_train.shape)
    y_train = tf.cast(x=y_train, dtype=tf.int32)
    y_train = tf.one_hot(indices=y_train, depth=10)

    return x_train, y_train


db_val = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_val = db_val.map(train_preprocess).shuffle(1000)

model = tf.keras.models.load_model('number_net_cnn')


def predict(model, test_data):
    # print(test_data)
    res = model.predict(x=test_data)
    # print(res)
    res = tf.argmax(res[0], 0)
    print("预测值：", res)


import cv2 as cv
import numpy as np

#
m2png = cv.imread('dataset/h4.png', cv.IMREAD_GRAYSCALE)
# m2png = cv.resize(m2png, (28, 28))
array = np.array(m2png)

# print(array.shape)
array = tf.expand_dims(array, axis=2)
array = tf.expand_dims(array, axis=0)
array = 255. - tf.cast(array, tf.float32)
print('array shape', array.shape)
predict(model, array)
# cv.namedWindow('blue', cv.WINDOW_KEEPRATIO)
# cv.imshow('blue', array)
#
# code = cv.waitKey(0)
# cv.destroyAllWindows()
