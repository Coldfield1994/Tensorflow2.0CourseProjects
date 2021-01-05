# The main file of a beginner Tensorflow2.X tutorial project, performing a linear regression classification on Titanic passenger data.

from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from tensorflow import feature_column as fc
import tensorflow as tf


# Loading Datasets
TrainingData = pd.read_csv('http://storage.googleapis.com/tf-datasets/titanic/train.csv')
EvaluationData = pd.read_csv('http://storage.googleapis.com/tf-datasets/titanic/eval.csv')

# Separating Survival Data from Original Datasets
y_Training = TrainingData.pop('survived')
y_Evaluation = EvaluationData.pop('survived')

# Column Classification
CategoricalColumns = ['sex', 'parch', 'class', 'deck', 'embark_town', 'alone']
NumericalColumns = ['age', 'n_siblings_spouses', 'fare']

# Creating Feature Columns to Change Categorical Data to Numerals
FeatureColumns = []
for FeatureName in CategoricalColumns:
    Vocabulary = TrainingData[FeatureName].unique()
    FeatureColumns.append(tf.feature_column.categorical_column_with_vocabulary_list(FeatureName, Vocabulary))

for FeatureName in NumericalColumns:
    FeatureColumns.append(tf.feature_column.numeric_column(FeatureName, dtype=tf.float32))

# Creating Input Function
def MakeInputFunction(Data_DataFrame, Label_DataFrame, EpochNumber=10, Shuffle=True, BatchSize=32):
    def InputFunction():
        Dataset = tf.data.Dataset.from_tensor_slices((dict(Data_DataFrame),Label_DataFrame))
        if Shuffle:
            Dataset = Dataset.shuffle(1000)
        Dataset = Dataset.batch(BatchSize).repeat(EpochNumber)
        return Dataset
    return InputFunction

# Creating Training and Evaluation Functions
TrainInputFunction = MakeInputFunction(TrainingData, y_Training)
EvaluateInputFunction = MakeInputFunction(EvaluationData, y_Evaluation, EpochNumber=1, Shuffle=False)

# Create Linear Classifier Estimator
LinearEstimator = tf.estimator.LinearClassifier(FeatureColumns)

# Train and Evaluate Linear Classifier Estimator
LinearEstimator.train(TrainInputFunction)
Result = LinearEstimator.evaluate(EvaluateInputFunction)

# Print Accuracy Results
clear_output()
print(Result['accuracy'])


