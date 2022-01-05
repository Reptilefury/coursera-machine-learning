import numpy as np
import tensorflow as tf
from sklearn.datasets import load_boston
from tensorflow_estimator.python.estimator.canned.timeseries import model


tf.compat.v1.disable_eager_execution()
tf.compat.v1.reset_default_graph()

boston = load_boston()
features = np.array(boston.data)
labels = np.array(boston.target)

n_training_sample = features.shape[0]
n_dim = features.shape[1]


def normalise(dataset):
    mu = np.mean(dataset, axis=0)
    sigma = np.std(dataset, axis=0)
    return (dataset - mu) / sigma


train_x = np.transpose(features)
train_y = np.transpose(labels)

print(train_x.shape)
print(train_y.shape)

x = tf.compat.v1.placeholder(tf.float32, [n_dim, None])
y = tf.compat.v1.placeholder(tf.float32, [1, None])

learning_rate = tf.compat.v1.placeholder(tf.float32, shape=())

W = tf.Variable(tf.ones([n_dim, 1]))
b = tf.Variable(tf.zeros(1))

init = tf.compat.v1.global_variables_initializer()
y_ = tf.matmul(tf.transpose(W), x) + b
cost = tf.reduce_mean(tf.square(y_, y))

training_step = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(cost)


def runlinearmodel(learning_r, training_epochs, train_obs, train_labels, debug=False):
    sess = tf.compat.v1.Session()
    sess.run(init)
    cost_history = np.empty(shape=[0], dtype=float)
    with tf.compat.v1.Session() as sess:
        for epoch in range(training_epochs + 1):
            sess.run(training_step, feed_dict={x: train_obs, y: train_labels, learning_rate: learning_r})
            cost_history = np.append(cost_history, cost)
            if (epoch % 1000 == 0) & debug:
                print("Reached epoch", epoch, "Cost J =", str.format('{0:.6f}', cost))
                return sess, cost_history
            sess, cost_history = runlinearmodel(learning_r=0.01, training_epochs=10000, train_obs=train_x,
                                                train_labels=train_y, debug=True)
