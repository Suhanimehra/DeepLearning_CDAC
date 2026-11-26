### Classification of Network Topologies
- 1. Fully Connected Networks (FCNs)
- 2. Convolutional Neural Networks (CNNs)
- 3. Recurrent Neural Networks (RNNs)

### Fully Connected Networks (FCNs)
- Every neuron in one layer links to every neuron in the next layer.
<img src="image.png" alt="Project Logo" width="400" height = "150">

- **Characteristics:**
  - **Density:** Many parameters, so compute cost is high.
  - **Uniformity:** No built-in assumptions about feature layout; treats inputs equally.
- **Applications:**
  - General tasks like text classification and image recognition.
  - Often used as final classification layers after CNN feature extraction.
- **Limitations:**
  - High parameter counts risk overfitting on small datasets.
  - No built-in handling for sequences or spatial structure.

### 2. Convolutional Neural Networks (CNNs)
Built for grid-like data (such as images) with local spatial patterns.

<img src="image-1.png" alt="Project Logo" width="400" height = "150">

- **Characteristics:**
  - **Convolutional Layers:** Scan inputs with filters to extract local and hierarchical features with fewer parameters.
  - **Pooling Layers:** Downsample spatial size while keeping key features.
  - **Parameter Sharing:** Reuse the same filter across the input, cutting parameters and aiding shift-invariant detection.
- **Applications:**
  - Image and video recognition.
  - Grid-like signals such as spectrograms or speech.
- **Limitations:**
  - Standard CNNs do not model time or sequence order directly.

### 3. Recurrent Neural Networks (RNNs)
Built for sequential data using state carried across steps.


<img src="image-2.png" alt="Project Logo" width="400" height = "150">

- **Characteristics:**
  - **Feedback Loops:** Recurrent links pass prior step state forward as memory.
  - **Variable Sequence Length:** Handles input and output sequences of changing length.
- **Applications:**
  - Speech recognition, NLP, and time series forecasting.
  - Sequence-to-sequence tasks such as translation.
- **Limitations:**
  - Susceptible to vanishing or exploding gradients during training.
  - Struggles with long-range dependencies without gating variants like LSTM(Long Short-Term Memory).