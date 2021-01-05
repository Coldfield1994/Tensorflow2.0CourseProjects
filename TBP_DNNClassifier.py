# The main file of a beginner Tensorflow2.X tutorial project, performing a classification on three classes of flower.

from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from tensorflow import feature_column as fc
import tensorflow as tf


# Loading Datasets
CSV_ColumnNames = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SpeciesName = ['Setosa', 'Versicolor', 'Virginica']

TrainingDataPath = tf.keras.utils.get_file(
    "iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
TestingDataPath = tf.keras.utils.get_file(
    "iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

TrainingData = pd.read_csv(TrainingDataPath, names=CSV_ColumnNames, header=0)
TestingData = pd.read_csv(TestingDataPath, names=CSV_ColumnNames, header=0)

y_TrainingData = TrainingData.pop('Species')
y_TestingData = TestingData.pop('Species')

# Creating Input Function
def InputFunction(Features, Labels, Training=True, BatchSize=256):
    Dataset = tf.data.Dataset.from_tensor_slices((dict(Features), Labels))
    if Training:
        Dataset = Dataset.shuffle(1000).repeat()
    return Dataset.batch(BatchSize)


# Creating Feature Columns
FeatureColumns = []
for key in TrainingData.keys():
    FeatureColumns.append(tf.feature_column.numeric_column(key=key))

print(FeatureColumns)

# Building DNN Model: Using Two Hidden Layers of 30 and 10 Nodes, Respectively.
Classifier = tf.estimator.DNNClassifier(
    feature_columns=FeatureColumns,
    hidden_units=[30, 10],
    n_classes=3)


# Training DNN Model
Classifier.train(
    input_fn=lambda: InputFunction(TrainingData, y_TrainingData, Training=True),
    steps=5000)

# Testing DNN Model
Result = Classifier.evaluate(
    input_fn=lambda: InputFunction(TestingData, y_TestingData, Training=False))
print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**Result))


# DNN Model Prediction
def InputFunction(Features, BatchSize=256):
    return tf.data.Dataset.from_tensor_slices(dict(Features)).batch(BatchSize)


PredictionFeatures = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
Predict = {}

# Input Prompt
print("Please type numeric values as prompted:")
for PredicitonFeature in PredictionFeatures:
    valid = True
    while valid:
        val = input(PredicitonFeature + ": ")
        if not val.isdigit(): valid = False

    Predict[PredicitonFeature] = [float(val)]

# Run Prediction
PredictionResults = Classifier.predict(
    input_fn=lambda: InputFunction(Predict))

for PredictionDictionary in PredictionResults:
    ClassID = PredictionDictionary['class_ids'][0]
    Probability = PredictionDictionary['probabilities'][ClassID]

    print('Prediction is "{}" ({:.1f}%)'.format(
        SpeciesName[ClassID], 100*Probability))


