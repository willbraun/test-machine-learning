from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import pandas as pd

CSV_COLUMN_NAMES = ['SepalLength', 'Sepalwidth", "PetalLength", "Petalwidth", "Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']
# defining constants to help later on

train_path = tf.keras.utils.get_file(
    "iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
test_path= tf.keras.utils.get_file(
    "iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")
train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
#Here we use keras (a module inside of TensorFlow) to grab our datasets and read them into a pandas dataframe

print(train.head())

train_y = train.pop('Species')
test_y = train.pop('Species')

print(train.head())
print(train.shape)