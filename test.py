import matplotlib

import tensorflow as tf
import numpy as np
import timeit
from tensorflow.keras import datasets
import tensorflow.keras
import matplotlib.pyplot as plt

# matplotlib.use('TkAgg')
# A = tf.constant([[1, 2], [3, 4]])
# B = tf.constant([[5, 6], [7, 8]])
# C = tf.matmul(A, B)
#
# print(C)

# a = tf.random.uniform(shape=())
# print(a)

# b = tf.ones(shape=(2, 3))
# c = tf.constant([2, 3], dtype=tf.int32)
# d = tf.constant([4, 5])
#
# e = tf.add(c, d)
# print(c, e)
# c[0] = 10
# d[0] = 100
# print(c)
# print(d)
#
# print(b.dtype)
# print(b.shape)
# print(b.numpy())

# 求导
# x = tf.Variable(initial_value=3.)
# with tf.GradientTape() as tape:
#     y = tf.pow(x, 6)
# y_grad = tape.gradient(y, x)
# print(y_grad)


# 向量求梯度
# a = tf.constant([[1, 2, 3], [5, 6, 7]])
# print(tf.reduce_sum(a, 0))  # 0代表按列求和
# print(tf.reduce_sum(a, 1))  # 1代表按行求和

# 计算时间
# n = 200000
# a = tf.random.normal([1, n])
# b = tf.random.normal([n, 1])
#
#
# def cpu_run():
#     with tf.device('/cpu:0'):
#         c = tf.matmul(a, b)
#     # print(c)
#
#
# def gpu_run():
#     with tf.device('/gpu:0'):
#         tf.matmul(a, b)
#
#
# timeit.timeit(cpu_run, number=10)
# cputime = timeit.timeit(cpu_run, number=10)
# print(cputime)
#
# timeit.timeit(gpu_run, number=10)
# gputime = timeit.timeit(gpu_run, number=10)
# print(gputime)


# 算梯度
# a = tf.Variable(1.)
# b = tf.Variable(2.)
# c = tf.Variable(3.)
# w = tf.Variable(4.)
# with tf.GradientTape() as tape:
#     y = a * w ** 2 + b * w + c
# y_grad = tape.gradient(y, w)
# print(y_grad)


########################计算 start

# import numpy as np
#
# X_raw = np.array([2013, 2014, 2015, 2016, 2017], dtype=np.float32)
# y_raw = np.array([12000, 14000, 15000, 16500, 17500], dtype=np.float32)
#
# X = (X_raw - X_raw.min()) / (X_raw.max() - X_raw.min())
# y = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min())
#
# a, b = 0, 0
# 手动计算
# num_epoch = 10000
# learning_rate = 5e-4
# for e in range(num_epoch):
#     # 手动计算损失函数关于自变量（模型参数）的梯度
#     y_pred = a * X + b
#     grad_a, grad_b = 2 * (y_pred - y).dot(X), 2 * (y_pred - y).sum()
#
#     # 更新参数
#     a, b = a - learning_rate * grad_a, b - learning_rate * grad_b
#
# print(a, b)
# # tensorflow计算
# X = tf.constant(X)
# y = tf.constant(y)
# a = tf.Variable(0.)
# b = tf.Variable(0.)
# variables = [a, b]
# optimizer = tf.keras.optimizers.SGD(5e-4)
# for e in range(num_epoch):
#     with tf.GradientTape() as tape:
#         y_result = a * X + b
#         loss = (y - y_result) ** 2
#     grads = tape.gradient(loss, variables)
#     optimizer.apply_gradients(grads_and_vars=zip(grads, variables))
#
# print(variables)
########################计算end

a = np.array([1, 2, 3])
b = tf.constant('Hello tensorflow')
c = tf.constant(True)
d = tf.constant(12.)
# print(tf.cast(d, tf.float64).dtype)
e = tf.constant([1, 2, 0, -1])
# print(tf.cast(e, tf.bool))
# print(tf.zeros([]))
# print(tf.zeros([1]))
# print(tf.zeros([1, 3]))
# print(tf.fill([1, 5], 100))
# print(tf.random.normal([2, 3], 1, 1))
# print(tf.random.uniform([1, 2], 0, 10, tf.int32))

# a = tf.constant([[1, 2], [3, 4]])
# print("original:", a)
# b = tf.expand_dims(a, 0)
# print("after op:", b)
# print(tf.transpose(tf.constant([[1, 2], [3, 4]]), perm=[1, 0]))

# a = tf.constant([1, 0], dtype=tf.int32)
# b = tf.constant([[0.5, 0.5], [1., 0.]], dtype=tf.float32)
# # b = tf.random.normal(shape=[6, 6])
# a = tf.one_hot(a, depth=2)
# c = a[0]
# print('a[0]:', a)
# print('b[0]', b)
# loss = tf.keras.losses.categorical_crossentropy(a, b)
# print(loss)

y_true = [[0, 1, 0], [0, 0, 1]]
y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
print(loss.numpy())

#
# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(units=512, activation=tf.nn.relu),
#     tf.keras.layers.Dense(units=256, activation=tf.nn.relu),
#     tf.keras.layers.Dense(units=128, activation=tf.nn.relu),
#     tf.keras.layers.Dense(units=32, activation=tf.nn.relu),
#     tf.keras.layers.Dense(units=10),
# ])
#
# model.build(input_shape=[None, 28 * 28])
# model.summary()
# print(model.trainable_variables, model.trainable, model.variables, model.optimizer)

# z = tf.random.normal([2, 10])  # 构造输出层的输出
# y_onehot = tf.constant([1, 3])  # 构造真实值
# y_onehot = tf.one_hot(y_onehot, depth=10)  # one-hot 编码
# # 输出层未使用 Softmax 函数,故 from_logits 设置为 True
# # 这样 categorical_crossentropy 函数在计算损失函数前,会先内部调用 Softmax 函数
# loss = tf.losses.categorical_crossentropy(y_onehot, z, from_logits=True)
# print(loss)
# print(z)
# print(y_onehot)
