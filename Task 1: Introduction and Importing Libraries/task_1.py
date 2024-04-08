'''In this project, we will be working on building two neural network models using the popular MNIST dataset. The MNIST dataset consists of 60,000 examples of handwritten digit images for training and 10,000 examples for testing. Each image is a grayscale 28x28 pixel representation of a handwritten digit from 0 to 9.'''

import numpy as np

from tensorflow.keras.datasets import mnist
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping, LambdaCallback
from tensorflow.keras.utils import to_categorical

%matplotlib inline
