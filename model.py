import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

import cv2 as cv
import matplotlib.pyplot as plt

import gc
import os
from glob import glob

from sklearn.model_selection import train_test_split

# Hyperparameters
IMG_SIZE = 128
SPLIT = 0.2
EPOCHS = 10
BATCH_SIZE = 64

def loadData():
    lungImagePath = 'lung_image_sets'
    categories = os.listdir(lungImagePath) # Different classifications of lung cells

    numpyImages = []
    oneHotLabels = []

    # We want to load all of the images into a numpy array that correspondes to a one-hot encoded label stored as a numpy array 
    for index, type in enumerate(categories):
        # Get all images of that type
        images = glob(f'{lungImagePath}/{type}/*.jpeg')

        # Load the images into a numpy array
        for image in images:

            imageArr = cv.resize(cv.imread(image), (IMG_SIZE, IMG_SIZE))

            # Normalize the imageArr
            imageArr = imageArr / 255.0

            # Append the images to a list of all of the images 
            numpyImages.append(imageArr)
            # Append the encoded label to a list of the proper label for this image
            # oneHotLabels[i] = the index of categories that numpyImages[i] belongs to
            oneHotLabels.append(index)

    # Convert the lists to numpy arrays
    numpyImages = np.array(numpyImages)

    # Do the oneHotEncoding, because right now this the labels are an array of indices
    oneHotLabels = pd.get_dummies(oneHotLabels).values

    return numpyImages, oneHotLabels

def buildModel():
    model = keras.Sequential()
    model.layers.add(layers.Conv2D(filters=16, kernel_size=(5, 5), activation='sigmoid', input_shape=(IMG_SIZE, IMG_SIZE, 3)))
    model.layers.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.layers.add(layers.Conv2D(filters=32, kernel_size=(5, 5), activation='sigmoid'))
    model.layers.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.layers.add(layers.Flatten())
    model.layers.add(layers.Dense(256, activation='sigmoid'))
    model.layers.add(layers.BatchNormalization)
    model.layers.add(layers.Dropout(0.5))
    model.layers.add(layers.Dense(64, activation='sigmoid'))
    model.layers.add(layers.BatchNormalization)
    model.layers.add(layers.Dropout(0.5))
    model.layers.add(layers.Dense(3, activation='softmax'))

    return model

def main():
    # Load the data
    images, labels = loadData()

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=SPLIT)

    # Build the model
    model = buildModel()

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])