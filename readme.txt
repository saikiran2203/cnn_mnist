 Handwritten Digits Recognition with CNN

 Overview

This project implements a Convolutional Neural Network (CNN) model for recognizing handwritten digits using the MNIST dataset. The project includes scripts for training and testing the model, as well as a pre-trained model file.

 Project Structure


HandWritten nums/
├── .dist/                 Directory containing additional resources or dependencies
├── mnist_cnn_model.h5     Pre-trained CNN model
├── train_mnist.py         Script to train the model
├── test_mnist.py          Script to test the model on new data


 Requirements

- Python 3.x
- TensorFlow
- Keras
- NumPy
- Matplotlib

Install the dependencies using:
bash
pip install tensorflow keras numpy matplotlib


 Usage

 1. Training the Model
To train the CNN model using the MNIST dataset:
bash
python train_mnist.py

The script saves the trained model as `mnist_cnn_model.h5`.

 2. Testing the Model
To test the pre-trained model on new data:
bash
python test_mnist.py


 3. Pre-trained Model
The file `mnist_cnn_model.h5` contains a pre-trained model for immediate testing.

 Dataset

The project uses the MNIST dataset, a benchmark dataset for handwritten digit recognition. It consists of 60,000 training images and 10,000 test images.

 Features

- CNN-based model for accurate handwritten digit recognition.
- Ready-to-use pre-trained model.
- Scripts for training and testing.

 Results

The model achieves high accuracy on the MNIST dataset, demonstrating its effectiveness for handwritten digit classification.

 Contributions

Contributions to improve the model or add new features are welcome. Feel free to fork the repository and submit a pull request.

