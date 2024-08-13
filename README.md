# DL Model for Lung Cell Classification

** Dataset from [Kaggle](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images/data) ** 

Implements a deep learning model that uses a Convolutional Neural Network to identify and classify lung tissue from histopathical images.
The data set includes 3 class of tissue with 5000 images for each class.
 - Benign Lung Tissue
 - Adenocarcinoma Lung Tissue
 - Squamous Cell Carcinoma Lung Tissue

The model currently achieves an 88.97% correct classification rate, with the following learning curves showing training verses validation loss and accuracy. 

![Figure_1](https://github.com/user-attachments/assets/a8689706-c826-414c-8888-cd004c88839c)
![Figure_2](https://github.com/user-attachments/assets/cd4d0e0d-d760-4908-8560-9a3ca768a00a)

## Table of Contents 
- [Model Design](#model-design)
- [Data Processing](#data-processing)

## Model Design
- The model uses a Convolution Neural Network with:
  -   2 Convolution Layers with MaxPooling
  -   2 Dense layers with BatchNormalization and Dropout of 0.25
  -   An output layer with 3 nodes that uses softmax to weigh probabilities.
- All layers besides the output use a 'relu' activation function.
- The model itself uses an ADAM optimizer with a lr of 0.005, categorical cross entropy for its loss function, and accuracy as its metric.

Below is a summary of the model:

| Layer (type)            | Output Shape        | Param #    |
|-------------------------|---------------------|------------|
| **conv2d (Conv2D)**          | (None, 124, 124, 16) | 1,216      |
| **max_pooling2d (MaxPooling2D)** | (None, 62, 62, 16)  | 0          |
| **conv2d_1 (Conv2D)**        | (None, 60, 60, 64)   | 9,280      |
| **max_pooling2d_1 (MaxPooling2D)** | (None, 30, 30, 64)  | 0          |
| **flatten (Flatten)**       | (None, 57600)      | 0          |
| **dense (Dense)**           | (None, 128)        | 7,372,928  |
| **batch_normalization (BatchNormalization)** | (None, 128)        | 512        |
| **dropout (Dropout)**       | (None, 128)        | 0          |
| **dense_1 (Dense)**         | (None, 32)         | 4,128      |
| **batch_normalization_1 (BatchNormalization)** | (None, 32)         | 128        |
| **dropout_1 (Dropout)**     | (None, 32)         | 0          |
| **dense_2 (Dense)**         | (None, 3)          | 99         |

**Total params:** 7,388,291 (28.18 MB)  
**Trainable params:** 7,387,971 (28.18 MB)  
**Non-trainable params:** 320 (1.25 KB)

## Data Processing
- The data is condensed using openCV's resize feature and then all pixels are normalized.
- The images are stored alongside a one-hot label that identifies each image's class.
- This data is then split into an 80/20 test/train split and a batch size of 64 images. 
