# evaluate.py
import tensorflow as tf
from cnn_model import create_cnn_model

# Load and preprocess CIFAR-10 test dataset
_, (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
test_images, test_labels = test_images / 255.0, test_labels

# Create and compile the CNN model
model = create_cnn_model()
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"\nTest accuracy: {test_acc}")
