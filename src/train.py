# train.py
import tensorflow as tf
from cnn_model import create_cnn_model

# Load and preprocess CIFAR-10 dataset
(train_images, train_labels), _ = tf.keras.datasets.cifar10.load_data()
train_images, train_labels = train_images / 255.0, train_labels

# Create and compile the CNN model
model = create_cnn_model()
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=10)
