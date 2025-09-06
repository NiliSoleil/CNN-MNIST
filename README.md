# CNN-MNIST

Deep Learning Project: Handwritten Digit Classification

This repository contains a class project that implements and trains a Convolutional Neural Network (CNN) for the classification of handwritten digits from the MNIST dataset. The project covers the full deep learning workflow, from data handling and model architecture to hyperparameter tuning and model evaluation.

Project Objectives

The main goals of this project were to:

  **Data Preparation:** 📊 To properly load, transform, and prepare the standard MNIST dataset for a deep learning model.
  
  **Model Definition:** 🧠 To define a specific CNN architecture with key components like convolutional layers and batch normalization.
  
  **Hyperparameter Optimization:** ⚙️ To systematically find the optimal learning rate and analyze the model's fit (overfitting, underfitting, or good fit).
  
  **Model Training and Evaluation:** 🏋️ To train the model, save the best-performing version, and visualize its performance.

**Key Project Steps**

The project was completed in the following stages:

**1. Dataset Preparation**

  📚 The MNIST dataset, consisting of handwritten digits from 0 to 9, was loaded using the torchvision library.
  
  ✨ Appropriate transforms were applied for preprocessing and data augmentation, and the data was converted to tensors.
  
  📦 A DataLoader was used for creating mini-batches.
  
  🖼️ A batch of images was displayed to visualize the dataset.
  
**2. Neural Network Model Definition**

  🧠 A specific CNN architecture was defined as a function. It includes convolutional layers, with each followed by a batch normalization and a ReLU activation layer. A padding of 1 was used for all convolutional layers.
  
  ✅ The model's output was checked by feeding it with random data.
  
  🔢 The total number of parameters in the model was calculated.

**3. Device Selection**

  🚀 The model was moved to a GPU for accelerated training.

**4. Loss Function and Optimizer**

  📉 The Cross-Entropy loss function and the SGD optimizer were defined.

**5. Model Training and Hyperparameter Tuning**

  Custom training and testing functions were defined as per the video tutorial.

  Forward and Backward Pass Check: A small batch of data was used to check the forward pass, and a small subset of the training data was used to check the backward pass over 10 epochs.

  Optimal Learning Rate Selection: 🔍 The model was trained for one epoch with different learning rates (0.1, 0.01, 0.001, 0.0001) to find a suitable starting value.

  Small Grid Search: A more precise learning rate and weight decay value were determined by training the network for 5 epochs and performing a grid search around the initial learning rate.

  Final Training: 📈 The model was then trained for a larger number of epochs using the optimal hyperparameters found in the previous steps.

**6. Results**

  📊 The learning curve (loss and accuracy over epochs) for the fully trained model was plotted.

  🔍 The model's learning curve was analyzed to confirm a good fit (no underfitting or overfitting). If the model was overfitting, techniques discussed in the lecture were used to fix the issue.

**7. Model Saving**

  💾 The fully trained model was saved to a file.

**How to Run the Code**

  Prerequisites: 🛠️ Make sure you have the necessary libraries, such as PyTorch, installed.

  Run the Notebook: 🖥️ The code is available here: https://github.com/NiliSoleil/CNN-MNIST/blob/main/CNN_MNIST.ipynb
  You can run it in Google Colab or any Jupyter environment.

Niloufar Soleil
