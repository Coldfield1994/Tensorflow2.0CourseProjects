# The main file of a beginner Tensorflow2.X tutorial project, predicting future weather based on probabilities.

from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from tensorflow import feature_column as fc
import tensorflow as tf
import tensorflow_probability as tfp


# Creating Probability Distributions

InitialDistribution = tfp.distributions.Categorical(probs=[0.8, 0.2])
TransitionDistribution = tfp.distributions.Categorical(probs=[[0.7, 0.3], [0.2, 0.8]])
ObservationDistribution = tfp.distributions.Normal(loc=[0., 15.], scale=[5., 10.])

# Creating Hidden Markov Model
HM_Model = tfp.distributions.HiddenMarkovModel(
    initial_distribution = InitialDistribution,
    transition_distribution = TransitionDistribution,
    observation_distribution = ObservationDistribution,
    num_steps = 7)

# Printing Results
mean = HM_Model.mean()

print(mean.numpy())