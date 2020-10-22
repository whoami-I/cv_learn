import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, datasets, Sequential, optimizers, metrics

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
print("x_test", x_test.shape, "y_test", y_test.shape)


# 数据预处理
def train_preprocess(x_train, y_train):
    # x_train = tf.cast(x=x_train, dtype=tf.float32) / 255.
    x_train = tf.cast(tf.reshape(x_train, [-1]), dtype=tf.float32)
    y_train = tf.cast(x=y_train, dtype=tf.int32)
    y_train = tf.one_hot(indices=y_train, depth=10)

    return x_train, y_train


db_val = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_val = db_val.map(train_preprocess).shuffle(1000)

model = tf.keras.models.load_model('number_net2')
xtest = tf.reshape(x_test, shape=[-1, 28 * 28])


# for step, (x_val, y_val) in enumerate(db_val):
#     print(x_val)
#     res = model.predict(x=tf.reshape(x_val, shape=(-1, 28 * 28)), batch_size=28 * 28)
#     print(res, res.shape)
#     res = tf.argmax(res[0], 0)
#     print("预测值：", res)
#     # real = tf.nn.softmax(y_val)
#     real = tf.argmax(y_val, 0)
#     print("真实值", real)
#     exit()
# res = model.predict(db_val)
# print(res)


def predict(model, test_data):
    # print(test_data)
    res = model.predict(x=test_data, batch_size=28 * 28)
    print(res)
    res = tf.argmax(res[0], 0)
    print("预测值：", res)


converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir='number_net2')
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
# import cv2 as cv
# import numpy as np
# #
# m2png = cv.imread('dataset/h4.png', cv.IMREAD_GRAYSCALE)
# # m2png = cv.resize(m2png, (28, 28))
# array = np.array(m2png)
# print(array)
# # print(array.shape)
# array = tf.reshape(array, (-1, 28 * 28))
# array = 255. - tf.cast(array, tf.float32)
# predict(model, array)
# cv.namedWindow('blue', cv.WINDOW_KEEPRATIO)
# cv.imshow('blue', array)
#
# code = cv.waitKey(0)
# cv.destroyAllWindows()
