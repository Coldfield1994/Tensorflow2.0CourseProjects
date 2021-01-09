# The main file of a beginner Tensorflow2.X tutorial project, performing a convolutional classification of ten objects from images using Keras.

from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from tensorflow import feature_column as fc
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models


# Loading Datasets
(TrainingImages, TrainingLabels), (TestingImages, TestingLabels) = datasets.cifar10.load_data()

# Normalizing Pixel Values
TrainingImages = TrainingImages/255
TestingImages = TestingImages/255

# Setting Classifiers
ClassNames = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Creating Model
Model = models.Sequential()
# Feature Extraction
Model.add(layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (32, 32, 3)))
Model.add(layers.MaxPooling2D((2, 2)))
Model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
Model.add(layers.MaxPooling2D((2, 2)))
Model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
# Dense Layer Classifier
Model.add(layers.Flatten())
Model.add(layers.Dense(64, activation='relu'))
Model.add(layers.Dense(10))

# Compile Model
Model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Evaluating Model
History = Model.fit(TrainingImages, TrainingLabels, epochs = 6, validation_data = (TestingImages, TestingLabels))





