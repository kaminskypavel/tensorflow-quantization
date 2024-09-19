import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

# Load and preprocess the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
test_images = test_images.reshape((10000, 28, 28, 1)).astype("float32") / 255

# Load the quantized model
interpreter = tf.lite.Interpreter(model_path="models/mnist_quantized_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the quantized model with a test image
test_image = test_images[0].reshape(1, 28, 28, 1).astype(np.float32)

# Set input tensor
interpreter.set_tensor(input_details[0]["index"], test_image)

# Run inference
interpreter.invoke()

# Get the output tensor
output_data = interpreter.get_tensor(output_details[0]["index"])
predicted_label = np.argmax(output_data)

print(f"Predicted Label: {predicted_label}, True Label: {test_labels[0]}")
