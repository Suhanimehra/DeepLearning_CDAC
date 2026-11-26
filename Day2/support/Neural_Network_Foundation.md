# Objective: 
1. Understanding Neural Networks Foundation
2. Distinguishing network layers
3. Master Network Topologies
4. Grasp the feed-forward mechanism
5. Comprehend backpropagation
6. Implement Enhanced Learning Techniques


1. Understanding Neural Networks Foundation
- Neural Network : A computational model inspired by the human brain, consisting of interconnected nodes (neurons) organized in layers. 
- Their operational principles mirror the way neurons in the brain process and transmit information. 
- It is used to approximate complex functions and patterns in data.

### The Neuron: Building Block of ANNs : which 
- It is a computational approximation of a biological neuron. 
-Disect the components of a neuron:
    - Inputs: Signals received from other neurons or external sources.
    - Activation Function: Introduces non-linearity and determines the neuron's output.
        - detail : After processing its input, a neuron produces an output by passing the cumulative input through the activation function.
        - common activation functions include 
            - Sigmoid : Maps input to a value between 0 and 1.
                - use scenarios: Often used in the output layer for binary classification tasks.
            - Tanh(Hyperbolic Tangent) : Maps input to a value between -1 and 1.
                - use scenarios: Often used in hidden layers of neural networks for capturing complex patterns and relationships in the data.
            - ReLU(Rectified Linear Unit) : Outputs the input directly if positive, otherwise outputs 0.
                - use scenarios: Commonly used in hidden layers of deep neural networks for introducing non-linearity and mitigating the vanishing gradient problem.
    - output: The final result produced by the neuron after applying the activation function.

Functionality
- Aggregation : Combines the weighted inputs received by the neuron.
- Transformation : Post-aggregation, the total is fed into the activation function to introduce non-linearity and produce the neuron's output.


## Synapses and Weights: Connections and Strength of Neural Networks

- **Synapses**: Represent the connections between neurons, allowing the transfer of signals.
- Weights: Parameters associated with each synapse that determine the strength and influence of the input signal on the neuron's output.

- **Synaptic Weight**: Each connection, or synapse, between two neurons in a network is associated with a numerical value known as a weight(strength or influence or importance of the connection).
- **Adjustment**: The process of "learning" in ANNs revolves around adjusting these weights. Through techniques like backpropagation and optimization algorithms like gradient descent, the network tweaks these weights to minimise the difference between its predicted output and the actual target 
values. 

Importance:
- Modelling Relationships: Weights allow the network to model intricate relationships in data. The magnitude and sign (positive or negative) of a weight can signify the kind and strength of the relationship between two neurons 
- Storage of Knowledge: The knowledge of an ANN is stored in its weights.  Once trained, the network's ability to generalise or make predictions on new data is a direct result of the patterns captured within these weights. 