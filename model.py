import numpy as np
import pandas as pd
import pickle

import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.callbacks import EarlyStopping

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

# Use pickle to save data to a file, so it does not have to be loaded every time
def saveData(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def openData(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

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
    model = keras.models.Sequential([
        layers.Conv2D(filters=16, kernel_size=(5, 5), activation='sigmoid', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(filters=64, kernel_size=(4, 4), activation='sigmoid'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(filters=128, kernel_size=(3, 3), activation='sigmoid'),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Flatten(),

        layers.Dense(256, activation='sigmoid'),
        layers.BatchNormalization(),
        layers.Dropout(0.25),
        layers.Dense(64, activation='sigmoid'),
        layers.BatchNormalization(),
        layers.Dropout(0.25),

        layers.Dense(3, activation='softmax')
    ])

    return model

class callBack(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        # Save the model
        # name = 'model' + str(epoch) + '.keras'
        # self.model.save(name)

        if logs.get('accuracy') > 0.99:
            print("\nReached 99% accuracy so cancelling training!")
            self.model.stop_training = True

def main():
    data_file = 'data.pkl'
    
    if os.path.exists(data_file):
        # Load data from file
        images, labels = openData(data_file)
    else:
        # Load and process data
        images, labels = loadData()
        # Save data to file
        saveData((images, labels), data_file)
    
    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=SPLIT)

    # Build the model
    model = buildModel()

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    earlyStop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    training = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(x_test, y_test), callbacks=[earlyStop, callBack()])

    trainingData = pd.DataFrame(training.history)
    trainingData.loc[:, ['loss', 'val_loss']].plot()
    trainingData.loc[:, ['accuracy', 'val_accuracy']].plot()
    plt.show()

if __name__ == '__main__':
    main()
    