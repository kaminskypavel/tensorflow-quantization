import tensorflow as tf
from tensorflow.keras.datasets import mnist

# Load and preprocess the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
test_images = test_images.reshape((10000, 28, 28, 1)).astype("float32") / 255

# Load the trained model
model = tf.keras.models.load_model("models/mnist_cnn_model.h5")

# Evaluate the model
loss, accuracy = model.evaluate(test_images, test_labels, verbose=2)
print(f"Original Model Accuracy: {accuracy * 100:.2f}%")
