import tensorflow as tf
from tensorflow.keras import layers, models
import os


EPOCHS = 15

# Load and preprocess the MNIST dataset
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1)).astype("float32") / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype("float32") / 255


# Build a simple CNN model
def build_model():
    model = models.Sequential(
        [
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dense(10, activation="softmax"),
        ]
    )
    return model


# Create a model
model = build_model()

# Compile the model
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# TensorBoard callback for logging
log_dir = os.path.join("logs", "fit")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Train the model with TensorBoard callback
model.fit(
    train_images,
    train_labels,
    epochs=EPOCHS,
    batch_size=64,
    validation_data=(test_images, test_labels),
    callbacks=[tensorboard_callback],
)

# Save the model
MODELNAME = f"models/nist_cnn_model-epochs.h5"
model.save(MODELNAME)

print(f"Model trained and saved as '{MODELNAME}'.")
