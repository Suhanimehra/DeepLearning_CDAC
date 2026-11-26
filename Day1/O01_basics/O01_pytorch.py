'''
nvidia-smi : Check the status of NVIDIA GPU and its memory usage.

pytorch installation steps:
1. Check NVIDIA GPU status: nvidia-smi
2. Install PyTorch: pip install torch torchvision torchaudio
3. Verify installation: python -c "import torch; print(torch.__version__)"

4. python -c "import struct; print(struct.calcsize('P') * 8)" : Check the system architecture (32-bit or 64-bit).
'''

'''
what is PyTorch? 
- PyTorch is an open-source deep learning framework developed by Facebook's AI Research lab. 
- It provides a flexible and efficient platform for building and training neural networks, supporting both CPU and GPU computations.
- It is widely used in both academia and industry for research and production purposes due to its dynamic computation graph and ease of use.
'''

'''
what are tensors in PyTorch?
- Tensors are the fundamental data structures in PyTorch, similar to arrays or matrices in other numerical computing libraries.
- They can be used to represent inputs, outputs, and parameters of neural networks.
- PyTorch provides a wide range of tensor operations, including mathematical, logical, and indexing operations, which can be performed on both CPU and GPU.
- Tensors can also be easily converted between CPU and GPU, enabling efficient computation on different hardware.
'''

'''
Hello, this is a basic PyTorch script.
Each model has 1 input, and learns to extract formula for converting celsius to fahrenheit.
z = w.x + b
# where w is the weight, x is the input, and b is the bias.
F = 1.8 * C + 32
'''

import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# print torch version
print(torch.__version__)

# create a celsius tensor
celsius = torch.tensor(
    [[-20.0], [-10.0], [0.0], [10.0], [20.0], [25.0], [30.0], [37.0], [100.0]], 
                dtype=torch.float32, 
                device=device
                )

fahrenheit = torch.tensor(
    [[-4.0], [14.0], [32.0], [50.0], [68.0], [77.0], [86.0], [98.6], [212.0]],
    dtype=torch.float32,
    device=device
)

# 1 input feature -> 1 output value (learns weight w and bias b)
model = nn.Linear(in_features=1, out_features=1)
model.to(device) 

loss_fn = nn.MSELoss() # Mean Squared Error loss for regression tasks
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001) # Stochastic Gradient Descent optimizer

# Train the model
for epoch in range(1, 50000):  # number of training iterations
    optimizer.zero_grad() # 1. Clearing previous gradients
    # Forward pass: compute predicted y by passing x to the model
    predictions = model(celsius) # 2. forward pass

    # Compute and print loss
    loss = loss_fn(predictions, fahrenheit) # 3. compute loss (MSE)

    loss.backward() # 4. backpropogation (compute gradients)
    optimizer.step() # 5. update weights and bias

    if epoch % 1000 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss.item():.6f}")

# Inspect learned parameters
learned_w = model.weight.item()
learned_b = model.bias.item()
print(f"\nLearned equation: F = {learned_w:.4f} * C + {learned_b:.4f}")

# Test on an unseen value (40°C should be 104.0°F)
model.eval()
# torch.no_grad() context is used to disable gradient calculation for inference, which reduces memory usage and speeds up computations.
with torch.no_grad():
    test_celsius = torch.tensor([[40.0]], dtype=torch.float32, device=device)
    test_fahrenheit = model(test_celsius)
    print(f"Predicted Fahrenheit for 40°C: {test_fahrenheit.item():.2f}")