# Digit-recognizer

MNIST Digit Classification with TensorFlow
This repository contains a simple neural network implementation using TensorFlow and Keras to classify handwritten digits from the MNIST dataset.

📌 Overview
This project demonstrates how to:

Load and preprocess the MNIST dataset.

Build and train a basic neural network.

Evaluate the model's performance.

Make predictions on test images and visualize the results.

🚀 Technologies Used
Python 3.x

TensorFlow 2.x

NumPy

Matplotlib

🧩 Model Architecture
plaintext
Copy
Edit
Input Layer      : 28x28 image (flattened)
Hidden Layer     : Dense (128 neurons) + ReLU
Output Layer     : Dense (10 neurons) + Softmax
🛠️ Setup Instructions
Clone the repository

bash
Copy
Edit
git clone https://github.com/nakulsharma4118/digit-recognizer.git
cd mnist-digit-classification
Install dependencies
Make sure Python and pip are installed, then run:

bash
Copy
Edit
pip install tensorflow numpy matplotlib
Run the script

bash
Copy
Edit
python mnist_classifier.py
📊 Output
The model is trained for 5 epochs.

A random test image is selected, and its predicted digit is displayed along with the actual label.

A plot shows the digit and prediction result.

🔍 Example Output
plaintext
Copy
Edit
Epoch 1/5
1875/1875 [==============================] - 2s 1ms/step - loss: 0.2583 - accuracy: 0.9250
...
Predicted digit: 7, Actual digit: 7
<!-- Add actual image or replace with local sample -->

🧠 Learnings
Neural networks can achieve high accuracy on digit recognition tasks with minimal architecture.

TensorFlow provides an easy interface to load datasets and create models quickly.

📂 File Structure
bash
Copy
Edit
├── mnist_classifier.py   # Main Python script
├── README.md             # Project documentation
