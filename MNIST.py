import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets.mnist import load_data
from sklearn.datasets import fetch_openml
mnist = fetch_openml('MNIST original')
#mnist = tf.keras.datasets.mnist

#(X_train, y_train), (X_test, y_test) = mnist.load_data()


#mnist = load_data("MNIST original")

# mnist = load_data.read_data_sets("MNIST_data/", one_hot=True)
X, Y = mnist["data"], mnist["target"]

output = X.shape()
print(output)


def plot_digit(some_digit):
    some_digit_image = some_digit.reshape(28, 28)
    plt.imshow(some_digit_image, cmap=matplotlib.cm.binary, interpolation="nearest")
    plt.axis("off")
    plt.show()
    plot_digit(X[36003])

    X_train = X[np.any([Y == 1, Y == 2], axis=0)]
    Y_train = Y[np.any([Y == 1, Y == 2], axis=0)]
    X_train_normalised = X_train / 255.0

    X_train_tr = X_train_normalised.transpose()
    y_train_tr = Y_train.reshape(1, Y_train.shape[0])
