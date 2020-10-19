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


def test_preprocess(x_test, y_test):
    x_test = tf.cast(tf.reshape(x_test, [-1]), dtype=tf.float32) / 255.
    # x_test = tf.cast(x=x_test, dtype=tf.float32) / 255.
    y_test = tf.cast(x=y_test, dtype=tf.int32)

    return x_test, y_test


train_db = tf.data.Dataset.from_tensor_slices(tensors=(x_train, y_train))
train_db = train_db.map(map_func=train_preprocess).shuffle(buffer_size=1000).batch(batch_size=128)

test_db = tf.data.Dataset.from_tensor_slices(tensors=(x_test, y_test))
test_db = test_db.map(map_func=train_preprocess).batch(batch_size=128)

# 建立网络模型
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=512, activation=tf.nn.relu),
    tf.keras.layers.Dense(units=256, activation=tf.nn.relu),
    tf.keras.layers.Dense(units=128, activation=tf.nn.relu),
    tf.keras.layers.Dense(units=32, activation=tf.nn.relu),
    tf.keras.layers.Dense(units=10),
])
model.build(input_shape=[None, 28 * 28])
model.summary()


def main():
    # input para
    model.compile(optimizer=optimizers.Adam(lr=1e-2),
                  loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # run network
    model.fit(train_db, epochs=5, validation_data=test_db, validation_freq=1)
    model.save('number_net2', include_optimizer=True)


if __name__ == '__main__':
    main()
