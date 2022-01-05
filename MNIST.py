from tensorflow.keras.datasets.mnist import load_data


mnist = load_data("MNIST original")

# mnist = load_data.read_data_sets("MNIST_data/", one_hot=True)
X, Y = mnist["data"], mnist["target"]

output = X.shape()
print(output)
