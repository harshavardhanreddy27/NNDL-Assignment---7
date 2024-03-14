# NNDL-Assignment---7
Neural Networks and Deep Learning Assignment - 7

# Fashion-MNIST Image Classification using Convolutional Neural Networks

This repository contains a Python script for training a Convolutional Neural Network (CNN) model to classify fashion items in the Fashion-MNIST dataset.

## Dependencies
- Python 3.x
- NumPy
- Matplotlib
- Scikit-learn
- TensorFlow/Keras

## Dataset
The Fashion-MNIST dataset consists of grayscale images of fashion items such as T-shirts, trousers, coats, etc. Each image is 28x28 pixels.

## Model Architecture
The CNN model architecture consists of convolutional layers, max-pooling layers, flattening, and dense layers with ReLU activation for convolutional layers and softmax for the output layer.

## Training
The model is trained using the Adam optimizer and sparse categorical cross-entropy loss. It is trained for 10 epochs with a batch size of 32.

## Evaluation
The model's performance is evaluated on a separate testing set, and metrics such as accuracy and loss are computed.

## Visualization
Various visualizations, including the confusion matrix, training/validation loss and accuracy plots, are generated to analyze the model's behavior and performance.

## Usage
1. Ensure all dependencies are installed.
2. Run the provided Python script to train and evaluate the model.
3. Visualize the results and performance metrics.
4. Modify the script or model architecture as needed for further experimentation.

For more details, refer to the comments in the Python script.

