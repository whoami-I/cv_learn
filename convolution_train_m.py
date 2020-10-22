from tensorflow.keras import Sequential
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, datasets, Sequential, optimizers, metrics

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


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


def test_preprocess(x_test, y_test):
    x_test = tf.cast(tf.reshape(x_test, [-1]), dtype=tf.float32)
    # x_test = tf.cast(x=x_test, dtype=tf.float32) / 255.
    y_test = tf.cast(x=y_test, dtype=tf.int32)

    return x_test, y_test


train_db = tf.data.Dataset.from_tensor_slices(tensors=(x_train, y_train))
train_db = train_db.map(map_func=train_preprocess).shuffle(buffer_size=1000).batch(batch_size=4)

test_db = tf.data.Dataset.from_tensor_slices(tensors=(x_test, y_test))
test_db = test_db.map(map_func=train_preprocess).batch(batch_size=4)

model = Sequential([  # 网络容器
    layers.Conv2D(6, kernel_size=3, strides=1),  # 第一个卷积层, 6 个 3x3 卷积核
    layers.MaxPooling2D(pool_size=2, strides=2),  # 高宽各减半的池化层
    layers.ReLU(),  # 激活函数
    layers.Conv2D(16, kernel_size=3, strides=1),  # 第二个卷积层, 16 个 3x3 卷积核
    layers.MaxPooling2D(pool_size=2, strides=2),  # 高宽各减半的池化层
    layers.ReLU(),  # 激活函数
    layers.Flatten(),  # 打平层,方便全连接层处理
    layers.Dense(120, activation='relu'),  # 全连接层,120 个节点
    layers.Dense(84, activation='relu'),  # 全连接层,84 节点
    layers.Dense(10)  # 全连接层,10 个节点
])
model.build(input_shape=[4, 28, 28, 1])
model.summary()


def main():
    # input para
    model.compile(optimizer=optimizers.Adam(lr=1e-4),
                  loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # run network
    model.fit(train_db, epochs=20, validation_data=test_db, validation_freq=1)
    # model.save('number_net2', include_optimizer=True)
    tf.saved_model.save(model, 'number_net_cnn')


if __name__ == '__main__':
    main()
