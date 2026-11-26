# Objective:
1. Understanding Neural Networks Foundation
2. Distinguishing network layers
3. Master Network Topologies
4. Grasp the feed-forward mechanism
5. Comprehend backpropagation
6. Implement Enhanced Learning Techniques

1. Understanding Neural Networks Foundation
- **Neural Network:** Connected nodes (neurons) arranged in layers, loosely inspired by the brain, used to learn patterns in data.
- Works by passing signals from layer to layer and tuning internal settings until predictions improve.

### The Neuron: Building Block of ANNs : which
- A simple math unit that takes inputs, combines them, and produces one output.
- Main parts of a neuron:
  - **Inputs:** Incoming numbers from data or earlier neurons.
  - **Activation Function:** Decides if and how strongly a neuron fires by adding non-linearity (bending straight-line math so networks can learn curves and complex patterns).
    - **Detail:** Sums weighted inputs and passes the total through the activation function.
    - **Common activation functions:**
      - **Sigmoid:** Squashes numbers to `0` to `1`; great for yes/no (binary) output probabilities.
      - **Tanh (Hyperbolic Tangent):** Squashes numbers to `-1` to `1`; often used in hidden layers to center data around zero.
      - **ReLU (Rectified Linear Unit):** Keeps positive numbers as they are and sets negatives to `0`; fast and common in hidden layers to avoid vanishing gradients.
  - **Output:** The neuron's final value after the activation function.
---
| Activation | Formula | Output Range | Typical Placement | Main Trade-Off | When to Choose |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ReLU** | $\max(0, z)$ | $[0, \infty)$ | Hidden layers (CNNs, MLPs) | Fast and avoids vanishing gradients for $z > 0$, but neurons can stop updating ("dying ReLU") when $z \le 0$. | Default choice for hidden layers in deep networks when fast math and strong gradients on positive inputs matter. |
| **Sigmoid** | $\frac{1}{1 + e^{-z}}$ | $(0, 1)$ | Binary output layer, gating mechanisms | Interpretable as probability, but saturates near `0` and `1` and causes vanishing gradients in deep stacks. | Predicting binary class probabilities (`0` or `1`), multi-label probabilities, or `0`–`1` gates (not deep hidden layers). |
| **Tanh** | $\frac{e^z - e^{-z}}{e^z + e^{-z}}$ | $(-1, 1)$ | Hidden layers, RNN/LSTM gates & states | Zero-centered output often helps layer-to-layer flow better than Sigmoid, but still saturates at extreme values. | Zero-centered hidden activations or recurrent states where negative, zero, and positive outputs are useful. |
---
Functionality
- **Aggregation:** Adds up weighted inputs (plus bias).
- **Transformation:** Applies the activation function to that sum to get the output.

## Synapses and Weights: Connections and Strength of Neural Networks
- **Synapses:** Links that carry signals from one neuron to the next.
- **Weights:** Numbers on each link showing how much influence an input has.

- **Synaptic Weight:** The importance or strength of a specific connection between two neurons.
- **Adjustment:** Learning means tweaking weights (via backpropagation and gradient descent) so guesses get closer to the real answers.

Importance:
- **Modelling Relationships:** Positive/negative signs and sizes show push/pull and strength between signals.
- **Storage of Knowledge:** Learned patterns live inside the weights, letting the network predict on new data.

---
---
### Layers in Neural Networks
- Input Layer: Gateway to the Network (Entry point that passes input data into the network.)
    - Features:
        - **Number of Neurons**: Matches the input feature count (e.g., a 28×28 grayscale image = 784 features = 784 neurons).
        - **Data Normalisation**: Scaling data (such as min-max or z-score) helps training stay stable and fast.
        - **Role**: Receives raw inputs and forwards them in a usable form to hidden layers.

- Hidden Layers: Where Magic Happens
    - Features:
        - **Depth of the Network:**: More hidden layers increase depth and allow learning more abstract features (deep learning). 
        - **Activation Functions**: Add non-linearity so the network can learn complex patterns (e.g., ReLU, Sigmoid, Tanh).
        - **Weights & Biases**: Learnable parameters tuned during training to reduce prediction error.
        - **Role**: Distills signals into representations where classification or prediction becomes easier.

- Output Layer: Final Decisions and Predictions (Final layer that produces the model's prediction from hidden-layer features.)
    - Features:
        - **Number of Neurons**: Matches class count for classification, or 1 neuron (typically) for regression.
       - **Activation Functions**: Depends on the task—
            - Sigmoid for binary classification, 
            - Softmax for multi-class, and 
            - linear or none for regression.
        - **Role**: Combines learned features into final outputs such as probabilities, class scores, or continuous values.