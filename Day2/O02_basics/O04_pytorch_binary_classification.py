'''
We will use pytorch to build a binary classification model.
binary classification neuon (Pass=1 vs Fail=0 based on Hours studied)

Steps
1. Define explicit real-world data: hours studied (x ∈ R) and corresponding pass/fail labels (y ∈ {0,1})
2. Compute the linear combination : z = w * x + b
3. Apply a non-linear activation: σ(z) = 1 / (1 + exp(-z)) to map any real number into a probability-like output in (0, 1) for binary classification.`
4. Measure classification error: Use Binary Cross-Entropy loss to quantify the difference between predicted probabilities and actual labels.
5. Train via backpropagation and gradient descent: Update the model parameters (weights and bias) iteratively to minimize the loss function.
6. Threshold predictions into classes: Convert the predicted probabilities into binary class labels (0 or 1) using a threshold, typically 0.5.
'''

'''
what is binary cross entropy loss? 
Binary Cross-Entropy loss measures the difference between the predicted probability and the actual binary label (0 or 1). It is commonly used for binary classification tasks.
It is defined as: L = -[y * log(p) + (1 - y) * log(1 - p)], where y is the true label and p is the predicted probability.
'''


import torch
import torch.nn as nn

# native windows CPU settings
device = torch.device("cpu")
print("Using device:", device)
# pytorch version
print("PyTorch version:", torch.__version__)

# step 1 : Prepare the data
# Explicit data: Hours studied (input) -> Pass (1) or Fail (0)
hours_studied = torch.tensor(
        [[1.0], [1.5], [2.0], [2.5], [3.0], [4.5], [5.0], [6.0], [7.0], [8.0]],
        dtype=torch.float32,
        device=device,
    )
passed_exam = torch.tensor(
        [[0.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0], [1.0]],
        dtype=torch.float32,
        device=device,
    )

# step 2 : single neuron : z = w * x + b : prepare the model
model = nn.Linear(in_features=1, out_features=1).to(device)
# followed by sigmoid in BCEWithLogitsLoss during training
criterion = nn.BCEWithLogitsLoss() # Binary Cross-Entropy loss with logits (includes sigmoid)
# sigmoid : used to map the linear output to a probability between 0 and 1
optimizer = torch.optim.SGD(model.parameters(), lr=0.01) # Stochastic Gradient Descent optimizer
# parameters SGD : model.parameters() contains the weights and bias of the linear layer that will be updated during training, lr = learning rate for the optimizer
# The optimizer will adjust the model parameters to minimize the loss during training

# step 3 : Training loop
total_epochs = 3000
step_epochs = 600

for start_epoch in range(0, total_epochs, step_epochs):
    for _ in range(step_epochs):
        optimizer.zero_grad()  # Clear previous gradients
        logits = model(hours_studied)  # Forward pass: compute the linear output
        loss = criterion(logits, passed_exam)  # Compute the binary cross-entropy loss
        loss.backward()  # Backward pass: compute gradients
        optimizer.step()  # Update model parameters
    end_epoch = start_epoch + step_epochs
    print(f"Epoch {end_epoch:4d}/{total_epochs} | BCE Loss: {loss.item():.6f}")

w = model.weight.item()  # Extract the learned weight
b = model.bias.item()    # Extract the learned bias
boundary_hours = -b / w if w != 0 else float('nan')  # Compute the decision boundary (hours studied for 50% pass probability)
print(f"Decision boundary (probability = 0.50): ~{boundary_hours:.2f} hours")

# step 5 : Make predictions for new data
test_hours = torch.tensor([[1.8], [3.5], [4.0], [6.5]], dtype=torch.float32, device=device)
model.eval()
with torch.no_grad():
    logits = model(test_hours)
    probs = torch.sigmoid(logits).squeeze(dim=1)

print("\nTest Predictions:")
for hrs, prob in zip(test_hours.squeeze(dim=1).tolist(), probs.tolist()):
    label = "PASS (1)" if prob >= 0.5 else "FAIL (0)"
    print(f"  {hrs:4.1f} hrs -> Prob(Pass): {prob:.4f} | Predicted: {label}")

'''
summary of the model:
- The model is a single neuron (linear layer) followed by a sigmoid activation (handled by BCEWithLogitsLoss).
- The model learns to predict the probability of passing the exam based on hours studied.
- The decision boundary is computed as the number of hours studied for a 50% pass probability.
- The model is trained using Stochastic Gradient Descent (SGD) to minimize the binary cross-entropy loss.
- Predictions for new data are made by applying the learned linear transformation followed by the sigmoid function to obtain probabilities.
- The predicted label is determined based on whether the probability is greater than or equal to 0.5 (PASS) or less than 0.5 (FAIL).

Step-by-step summary:
- prepare the dataset with hours studied and corresponding pass/fail labels.
- define a single neuron model (linear layer) for binary classification.
- use BCEWithLogitsLoss as the loss function, which combines a sigmoid activation with binary cross-entropy loss.
- choose Stochastic Gradient Descent (SGD) as the optimizer to update model parameters.
- run the training loop for a specified number of epochs, computing the loss, performing backpropagation, and updating the model parameters.
- after training, extract the learned weight and bias to compute the decision boundary for a 50% pass probability.
- make predictions for new data by applying the model and sigmoid function to obtain probabilities.
- determine the predicted label based on whether the probability is greater than or equal to 0.5 (PASS) or less than 0.5 (FAIL).
'''