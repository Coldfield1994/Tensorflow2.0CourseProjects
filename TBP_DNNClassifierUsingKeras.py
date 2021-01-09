# The main file of a beginner Tensorflow2.X tutorial project, performing a classification of ten articles of clothing from images using Keras.

from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from tensorflow import feature_column as fc
import tensorflow as tf
from tensorflow import keras

# Loading Datasets
Fashion_MNIST = keras.datasets.fashion_mnist
(TrainingImages, TrainingLabels), (TestingImages, TestingLabels) = Fashion_MNIST.load_data()
ClassifierNames = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Data Preprocessing
TrainingImages = TrainingImages/255
TestingImages = TestingImages/255

# Building Model
Model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),        # Flatten = converts 2D to 1D
    keras.layers.Dense(128, activation = 'relu'),     # Dense = all neurons are connected to all previous layer neurons
    keras.layers.Dense(10, activation = 'softmax')
])

# Compile Model
Model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train Model
Model.fit(TrainingImages, TrainingLabels, epochs=10)

# Evaluating Model
TestingLoss, TestingAccuracy = Model.evaluate(TestingImages, TestingLabels, verbose = 1)
print('Test accuracy:', TestingAccuracy)

# Make Predictions
Predictions = Model.predict(TestingImages)
print(ClassifierNames[np.argmax(Predictions[2])])
plt.figure()
plt.imshow(TestingImages[2])
plt.colorbar()
plt.grid(False)
plt.show()
