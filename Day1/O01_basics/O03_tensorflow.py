'''
what is TensorFlow? 
- TensorFlow is an open-source deep learning framework developed by Google.
- It provides a flexible and efficient platform for building and training neural networks, supporting both CPU and GPU computations.
- It is widely used in both academia and industry for research and production purposes due to its computational graph and ease of use.
'''
'''
What is keras?
- Keras is an open-source deep learning API written in Python, running on top of TensorFlow.
- It provides a high-level interface for building and training neural networks, making it easier to experiment with deep learning models.
- Keras is known for its user-friendly and modular design, allowing for quick prototyping and experimentation.
'''
'''
Hello, this is a basic TensorFlow script.
Each model has 1 input, and learns to extract formula for converting celsius to fahrenheit.
z = w.x + b
# where w is the weight, x is the input, and b is the bias.
F = 1.8 * C + 32
'''

'''
installation:
- You can install TensorFlow using pip:
  ```
  pip install tensorflow
  ```
'''

'''
In PowerShell (ctrl + `), you can set the environment variables as follows:
$env:TF_CPP_MIN_LOG_LEVEL="2"
$env:TF_ENABLE_ONEDNN_OPTS="0"
'''
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2") # Suppress TensorFlow logging messages (INFO and WARNING) but show errors
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0") # Disable oneDNN optimizations for TensorFlow

import numpy as np
import tensorflow as tf



# $env:F_ENABLE_ONEDNN_OPTS=0 # Disable oneDNN optimizations for TensorFlow
# print TensorFlow version
print(tf.__version__)

# Check if GPU is available
print("TensorFlow version:", tf.__version__)
gpus = tf.config.list_physical_devices("GPU")
print("GPUs available:", len(gpus), "| Running on CPU" if not gpus else "| Running on GPU")

# Explicit Celsius (input) and Fahrenheit (target) pairs: F = 1.8 * C + 32
celsius = np.array(
        [-20.0, -10.0, 0.0, 10.0, 20.0, 25.0, 30.0, 37.0, 100.0],
        dtype=np.float32,
    ).reshape(-1, 1)

fahrenheit = np.array(
        [-4.0, 14.0, 32.0, 50.0, 68.0, 77.0, 86.0, 98.6, 212.0],
        dtype=np.float32,
    ).reshape(-1, 1)

# Single Dense neuron with 1 input and 1 output (linear activation by default)
model = tf.keras.Sequential([ # Sequential model with a single input and output layer
        tf.keras.Input(shape=(1,)),# Input layer with 1 feature (Celsius)
        tf.keras.layers.Dense(units=1),# Single neuron with 1 input and 1 output
    ])

# Compile the model with Stochastic Gradient Descent optimizer and Mean Squared Error loss function
model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.0001),
        loss="mean_squared_error",
    )

# Train in stages so you can see loss decreasing every 1,000 epochs
total_epochs = 5000
step_epochs = 1000
for start_epoch in range(0, total_epochs, step_epochs):
        history = model.fit(
            celsius,
            fahrenheit,
            epochs=step_epochs,
            verbose=0,
        )
        end_epoch = start_epoch + step_epochs
        final_loss = history.history["loss"][-1]
        print(f"Epoch {end_epoch:4d}/{total_epochs} | MSE Loss: {final_loss:.6f}")

# Inspect learned parameters
weights, bias = model.layers[0].get_weights()
learned_w = float(weights[0][0])
learned_b = float(bias[0])
print(f"\nLearned equation: F = {learned_w:.4f} * C + {learned_b:.4f}")
print("Target equation:  F = 1.8000 * C + 32.0000")

# Test on unseen Celsius values
test_temps = np.array([[15.0], [40.0], [75.0]], dtype=np.float32)
predictions = model.predict(test_temps, verbose=0).flatten()
print("\nTest Predictions:")
for c_val, pred_f in zip(test_temps.flatten(), predictions):
        expected_f = 1.8 * float(c_val) + 32.0
        print(f"  {c_val:5.1f}°C -> Predicted: {pred_f:6.2f}°F | Expected: {expected_f:6.2f}°F")