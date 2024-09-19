import time
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

# Load and preprocess the MNIST dataset
(_, _), (test_images, test_labels) = mnist.load_data()
test_images = test_images.reshape((10000, 28, 28, 1)).astype("float32") / 255

# Load the trained model
model = tf.keras.models.load_model("models/mnist_cnn_model.h5")


# Measure inference time
def measure_inference_time(model, test_images):
    start_time = time.time()
    model.predict(test_images, batch_size=64)  # Process in batches for better timing
    end_time = time.time()
    inference_time = end_time - start_time
    return inference_time


original_time = measure_inference_time(model, test_images)
print(f"Original Model Inference Time: {original_time:.2f} seconds")


# Load the quantized model

# Load and preprocess the MNIST dataset
(_, _), (test_images, test_labels) = mnist.load_data()
test_images = test_images.reshape((10000, 28, 28, 1)).astype("float32") / 255

# Load the quantized model
interpreter = tf.lite.Interpreter(model_path="models/mnist_quantized_model.tflite")
interpreter.allocate_tensors()


# Measure inference time
def measure_inference_time_tflite(interpreter, test_images):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    start_time = time.time()
    for image in test_images:
        input_data = np.expand_dims(image, axis=0).astype(np.float32)
        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()
    end_time = time.time()
    inference_time = end_time - start_time
    return inference_time


quantized_time = measure_inference_time_tflite(interpreter, test_images)
print(f"Quantized Model Inference Time: {quantized_time:.2f} seconds")
