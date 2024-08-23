# MNIST Handwritten Digit Classification using Deep Learning

## Overview

This project demonstrates the use of a simple deep neural network for classifying handwritten digits from the MNIST dataset. The MNIST dataset is a well-known benchmark in the field of machine learning and contains 60,000 training images and 10,000 test images of handwritten digits (0-9).

## Table of Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training the Model](#training-the-model)
- [Evaluation](#evaluation)
- [Results](#results)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Installation

To run this project, you need to install the necessary dependencies. You can do so by running the following command:

```bash
pip install numpy matplotlib seaborn opencv-python pillow tensorflow
```

If you are using Google Colab, most of these dependencies are pre-installed. You can use the provided code directly in a Colab notebook.

## Dataset

The MNIST dataset can be directly loaded from `keras.datasets`. The dataset is split into a training set of 60,000 images and a test set of 10,000 images. Each image is a grayscale image of size 28x28 pixels.

```python
from keras.datasets import mnist

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
```

## Model Architecture

The neural network used in this project consists of the following layers:

- **Input Layer:** Flattens the 28x28 image into a 784-dimensional vector.
- **Hidden Layers:** Two Dense layers with 50 neurons each and ReLU activation functions.
- **Output Layer:** Dense layer with 10 neurons (one for each digit) and a sigmoid activation function.

```python
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(50, activation='relu'),
    keras.layers.Dense(50, activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])
```

## Training the Model

The model is compiled using the Adam optimizer and the sparse categorical cross-entropy loss function. The training process runs for 10 epochs.

```python
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=10)
```

## Evaluation

The model is evaluated on the test set to determine its accuracy.

```python
loss, accuracy = model.evaluate(X_test, Y_test)
print(f"Test accuracy: {accuracy:.4f}")
```

## Results

After training, the model achieves an accuracy of approximately 97.1% on the test dataset.

## Usage

To use this code, simply clone the repository and run the provided notebook or script.

```bash
git clone https://github.com/vishesh711/mnist-digit-classification.git
cd mnist-digit-classification
```

Open the notebook and run the cells to train the model and evaluate it on the test dataset.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
