import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from keras import layers

import cv2 as cv
import matplotlib.pyplot as plt

import gc
import os
from glob import glob
# Hyperparameters
IMG_SIZE = 256
SPLIT = 0.2
EPOCHS = 10
BATCH_SIZE = 64

lungImagePath = 'lung_images_sets'
categories = os.listdir(lungImagePath) # Different classifications of lung cells

numpyImages = []
oneHotLabels = []

# We want to load all of the images into a numpy array that correspondes to a one-hot encoded label stored as a numpy array 
for index, type in categories:
    # Get all images of that type
    images = glob(f'{lungImagePath}/{type}/*.jpeg')

    # Load the images into a numpy array
    for image in images:
        imageArr = cv.resize(cv.imread(image), (IMG_SIZE, IMG_SIZE))

        # Append the images to a list of all of the images 
        numpyImages.append(imageArr)
        # Append the encoded label to a list of the proper label for this image
        # oneHotLabels[i] = the index of categories that numpyImages[i] belongs to
        oneHotLabels.append(index)

# Convert the lists to numpy arrays
numpyImages = np.array(numpyImages)

# Do the oneHotEncoding, because right now this the labels are an array of indices
oneHotLabels = pd.get_dummies(oneHotLabels).values
