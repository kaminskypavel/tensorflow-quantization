import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("mnist_cnn_model.h5")

# Convert the model to TensorFlow Lite with dynamic range quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()

# Save the quantized model
with open("mnist_quantized_model.tflite", "wb") as f:
    f.write(tflite_quant_model)

print("Quantized model saved as 'mnist_quantized_model.tflite'.")
