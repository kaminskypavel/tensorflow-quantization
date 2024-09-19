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


# Function to evaluate the model
def evaluate_model(test_images, test_labels):
    correct_predictions = 0
    for i in range(len(test_images)):
        # Prepare the input
        input_data = np.expand_dims(test_images[i], axis=0)
        interpreter.set_tensor(input_details[0]["index"], input_data)

        # Run inference
        interpreter.invoke()

        # Get the result
        output_data = interpreter.get_tensor(output_details[0]["index"])
        predicted_label = np.argmax(output_data)

        if predicted_label == test_labels[i]:
            correct_predictions += 1

    accuracy = correct_predictions / len(test_labels)
    return accuracy


# Evaluate the quantized model
quantized_accuracy = evaluate_model(test_images, test_labels)
print(f"Quantized Model Accuracy: {quantized_accuracy * 100:.2f}%")
