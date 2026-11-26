import os

# Keep console output clean on native Windows CPU
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

print("TensorFlow version:", tf.__version__)
gpus = tf.config.list_physical_devices("GPU")
print("Device available:", len(gpus), "| Running on CPU" if not gpus else "| Running on GPU")

    # Explicit data: Hours studied (input) -> Pass (1) or Fail (0)
hours_studied = np.array(
        [1.0, 1.5, 2.0, 2.5, 3.0, 4.5, 5.0, 6.0, 7.0, 8.0],
        dtype=np.float32,
    ).reshape(-1, 1)

passed_exam = np.array(
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        dtype=np.float32,
    ).reshape(-1, 1)

'''
whars are keras?
- Keras is a high-level API for building and training deep learning models. It provides a user-friendly interface to define neural networks, compile them with loss functions and optimizers, and train them on data. 
- Keras is integrated with TensorFlow, making it easy to leverage TensorFlow's capabilities while maintaining simplicity and readability in model development.    
'''

# Single neuron with Sigmoid activation for binary classification
model = tf.keras.Sequential([
        tf.keras.Input(shape=(1,)),
        tf.keras.layers.Dense(units=1, activation="sigmoid"),
    ])

# Compile the model with binary cross-entropy loss and an optimizer
model.compile(
    loss="binary_crossentropy", # Binary cross-entropy loss for binary classification tasks : measures the difference between predicted probabilities and actual binary labels : other options include "categorical_crossentropy" for multi-class classification
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.1), # Stochastic Gradient Descent optimizer with a learning rate of 0.1 : learning rate : controls the step size during parameter updates
    metrics=["binary_accuracy"] # Use binary accuracy for evaluating binary classification performance : other options are : "accuracy", "AUC", "Precision", "Recall"
)

total_epochs = 3000
step_epochs = 600

for start_epoch in range(0, total_epochs, step_epochs):
        history = model.fit(
            hours_studied,
            passed_exam,
            epochs=step_epochs, # Number of epochs to train in this step of the loop 
            verbose=0,
        )
        end_epoch = start_epoch + step_epochs
        loss_val = history.history["loss"][-1]
        acc_val = history.history["binary_accuracy"][-1]
        print(
            f"Epoch {end_epoch:4d}/{total_epochs} | "
            f"BCE Loss: {loss_val:.6f} | Accuracy: {acc_val:.2f}"
        )

weights, bias = model.layers[0].get_weights()
w = float(weights[0][0])
b = float(bias[0])
boundary_hours = -b / w if w != 0 else float("nan")
print(f"\nLearned parameters: w = {w:.4f}, b = {b:.4f}")
print(f"Decision boundary (probability = 0.50): ~{boundary_hours:.2f} hours")

# Test on unseen study hours
test_hours = np.array([[1.8], [3.5], [4.0], [6.5]], dtype=np.float32)
probs = model.predict(test_hours, verbose=0).flatten()

print("\nTest Predictions:")
for hrs, prob in zip(test_hours.flatten(), probs):
        label = "PASS (1)" if prob >= 0.5 else "FAIL (0)"
        print(f"  {float(hrs):4.1f} hrs -> Prob(Pass): {float(prob):.4f} | Predicted: {label}")