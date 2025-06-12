import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load and preprocess MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize pixel values

# Build a simple neural network
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile and train the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, verbose=1)

# Test the model on a random test image
test_index = 0
test_image = x_test[test_index]
prediction = model.predict(test_image[np.newaxis, ...])
predicted_digit = np.argmax(prediction)

# Display the test image with prediction
plt.imshow(test_image, cmap='gray')
plt.title(f'Predicted: {predicted_digit}, Actual: {y_test[test_index]}')
plt.axis('off')
plt.show()

print(f"Predicted digit: {predicted_digit}, Actual digit: {y_test[test_index]}")