import tensorflow as tf
converter = tf.lite.TFLiteConverter\
    .from_saved_model(saved_model_dir='number_net_cnn')
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

