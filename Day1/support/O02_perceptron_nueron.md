### Demystifying Perceptrons and Neurons
#### Perceptrons:
   - Foundational building block of neural networks and deep learning models.
   - It acts as a binary linear classifier, determining if an input belongs to one class or another.
   - Performs binary classification by computing a weighted sum of inputs and applying an activation function.
   - Introduced by Frank Rosenblatt in 1958.


#### Logic Gates Interpretation: 
  - Perceptrons can be used to model basic logic gates such as AND, OR, and NOT.
  - For instance, with appropriately adjusted weights, a perceptron can emulate the AND gate. Iftwo input values are '1' (true), the perceptron outputs '1' and '0' otherwise.
  - By adjusting the weights and bias, a perceptron can mimic the behavior of these gates.
  - This intrinsic capacity of perceptrons to reproduce logical operations underscores their poweras fundamental computational units in neural networks.

#### Linear Decision Boundaries:
- The perceptron makes its decision based on a linear function of the input. Essentially, if the weighted sum of the input surpasses a certain threshold, the perceptron activates; otherwise, it remains inactive. 
       
- In a 2D input space, this decision mechanism is represented as a line. For higher-dimensional inputs, it generalises to a hyperplane. 
- This line or hyperplane is termed the "decision boundary," delineating the regions corresponding to different output classes.


- Perceptrons are limited to linearly separable problems. If the data cannot be separated by a straight line (or hyperplane in higher dimensions), a single-layer perceptron will fail to classify it correctly. This limitation led to the development of multi-layer perceptrons (MLPs) which can handle non-linear decision boundaries through multiple layers and non-linear activation functions.

#### Perceptrons Layers:
- input layer: receives the input features.
- hidden layer(s): perform computations and extract features (can have multiple layers).
- output layer: produces the final prediction or output.

#### What is Perceptrons?
A **perceptron** is a type of artificial neuron used in machine learning for **binary classification** tasks.

#### Neurons:
  - Analogous to the perceptron, neurons compute a weighted sum of their inputs and pass the result through an activation function.
  - A neuron is the fundamental unit of a neural network, inspired by biological neurons.
  - It receives input signals, processes them using a weighted sum and an activation function, and produces an output signal.
  - Neurons can be organized into layers to form complex neural networks.

#### Perceptrons vs Neurons:
  - Perceptrons are a simplified model of neurons, primarily used for binary classification.
  - Neurons in modern neural networks are more flexible, using various activation functions and capable of handling complex, non-linear relationships.
  - While a single perceptron can only solve linearly separable problems, networks of neurons (multi-layer perceptrons) can approximate any continuous function given sufficient neurons and layers.

#### What are Connections and weights?
- Connections are the links between neurons in different layers of a neural network.
- Weights are numerical values assigned to these connections, determining the strength, Direction, and influence of the input signal on the neuron's output.
- Connections and weights are fundamental components of neural networks, enabling the network to learn and adapt by adjusting the weights during training.
- During Training, Weights are adjusted by using optimization techniques such as gradient descent to minimize the error between the predicted and actual outputs.
- This process allows the neural network to learn from data and improve its performance over time.

![alt text](image.png)

![alt text](image-1.png)

#### Heartbeat of Neurons:
#### Activation function:
  - An activation function is a mathematical function applied to the output of a neuron.
    - It introduces non-linearity into the network, allowing it to learn complex patterns.
    - Common activation functions include Sigmoid, Tanh, and ReLU (Rectified Linear Unit).
    - without these activation functions, neural networks would behave like linear models, limiting their ability to capture complex relationships in the data.
    - Non-Linearity :Allows neural networks to model complex relationships between inputs and outputs.
    - Thresholding:
        - Thresholding is a process where the output of a neuron is compared to a certain threshold value.
        - If the output exceeds the threshold, the neuron is activated; otherwise, it remains inactive.
        - This mechanism helps in controlling the firing of neurons and can be used to introduce non-linearity in the network.
    - Differentiability:
        - As deep learning models rely on gradient-based optimization techniques like gradient descent, the activation functions used should be differentiable.
        - Differentiability ensures that gradients can be computed for backpropagation, enabling the network to learn effectively.


#### Common Activation Functions:

- Sigmoid:
    - Maps input values to a range between 0 and 1.
    - Often used in the output layer for binary classification problems.
    - Suffers from the vanishing gradient problem, especially in deep networks.  This is because for very high or very low values of x, the gradient is almost zero.
    Equation: σ(x) = 1 / (1 + e^(-x))

- Tanh (Hyperbolic Tangent):
    - Maps input values to a range between -1 and 1.
    - Zero-centered, which can help with optimization.
    - Suffers from the vanishing gradient problem similar to Sigmoid.
    Equation: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))

- ReLU (Rectified Linear Unit):
    - Outputs the input directly if it is positive; otherwise, it outputs zero.
    - Widely used due to its simplicity and effectiveness in deep networks.
    - Helps mitigate the vanishing gradient problem, allowing for better training of deep networks.
    Equation: ReLU(x) = max(0, x)

- Leaky ReLU:
    - Similar to ReLU, but allows a small, non-zero gradient when the input is negative.
    - Helps address the "dying ReLU" problem where neurons can become inactive and stop learning.
    Equation: LeakyReLU(x) = max(0.01x, x)

### what is a bias?
- A bias is an additional parameter in a neuron that allows the activation function to be shifted to the left or right.
- It helps the model to fit the data better by providing more flexibility.
- Mathematically, for a neuron with input x and weight w, the output is calculated as:
  z = w * x + b
  where b is the bias term.


### what is backpropagation?

- Backpropagation is a supervised learning algorithm used for training artificial neural networks.
- It calculates the gradient of the loss function with respect to each weight by the chain rule, allowing the network to update the weights to minimize the loss.
- The process involves two main steps:
  1. Forward pass: Compute the output of the network and the loss.
  2. Backward pass: Propagate the error backward through the network to compute gradients and update weights.

- The essence of training a neural network lies in optimising its 
weights and biases to reduce the discrepancy between predicted 
and actual outcomes. 
Techniques used:
- Gradient Descent: 
- Stochastic Gradient Descent (SGD):
- Mini-batch Gradient Descent:
- Adam Optimizer:
- RMSprop:
- Adagrad:
- learning rate: 
- momentum: 
- chain Rule of calculus:  
- regularization:  





