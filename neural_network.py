import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt


fashion_mnist = keras.datasets.fashion_mnist # Load Fashion MNIST dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data() # split into training and testing
print(train_images.shape)
print(train_images[0, 23, 23])
print(train_labels[:10])

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal", "Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# show test image with pyplot
plt.figure()
plt.imshow(train_images[1])
plt.colorbar()
plt.grid(False)
plt.show()

# Data preprocessing to get data between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0