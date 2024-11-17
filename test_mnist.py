import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load the saved model
model = load_model('mnist_cnn_model.h5')

# Load the MNIST test dataset
_, (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Preprocess the test data
x_test = x_test / 255.0  # Normalize
x_test = x_test.reshape(-1, 28, 28, 1)  # Reshape

# Pick a random test image
random_idx = np.random.randint(0, len(x_test))
test_image = x_test[random_idx]
true_label = y_test[random_idx]

# Predict the digit
predicted_probs = model.predict(test_image.reshape(1, 28, 28, 1))
predicted_label = np.argmax(predicted_probs)

# Display the result
plt.imshow(test_image.reshape(28, 28), cmap='gray')
plt.title(f'True Label: {true_label}, Predicted: {predicted_label}')
plt.axis('off')
plt.show()
