import tensorflow as tf
# importing mnist dataset through Keras
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.python.ops.init_ops_v2 import normal
tf.compat.v1.disable_eager_execution()

mnist = load_data.read_data_sets("MNIST_data/", one_hot=True)

features_count = 784
labels_count = 10
batch_size = 128
epochs = 10
learning_rate = 0.5

features = tf.compat.v1.placeholder(tf.float32, [None, features_count])
labels = tf.compat.v1.placeholder(tf.float32, [None, labels_count])

weights = tf.Variable(tf.compat.v1.truncated, normal((features_count, labels_count)))
biases = tf.Variable(tf.zeros(labels_count), name='biases')


optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(loss)

logits = tf.add(tf.matmul(features, weights), biases)

prediction = tf.nn.softmax(logits)

cross_entrophy = tf.reduce_sum(labels * tf.compat.v1.log(prediction), reduction_indices=1)

loss = tf.reduce_mean(cross_entrophy)

init = tf.compat.v1.global_variables_initializer()

is_correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))

accuracy = tf.reduce_mean(tf.cast(is_correct_prediction, tf.float32))

with tf.compat.v1.Session() as sess:
    sess.run(init)
    total_batch = int(len(mnist.train.labels) / batch_size)
    for epoch in epochs:
        avg_cost = 0
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next.batch(batch_size),
            c = sess.run([optimizer, loss], feed_dict={features: batch_x, labels: batch_y})
            avg_cost += c / total_batch
            print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))
            print("Epoch:", (epoch + 1), "cost = ", "{:.3f}".format(avg_cost))
            print(sess.run(accuracy, feed_dict={features: mnist.test.images, labels: mnist.test.labels}))
