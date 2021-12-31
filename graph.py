import tensorflow as tf

tf.compat.v1.disable_eager_execution()

x1 = tf.compat.v1.placeholder(tf.float32, 1)
w1 = tf.compat.v1.placeholder(tf.float32, 1)
x2 = tf.compat.v1.placeholder(tf.float32, 1)
w2 = tf.compat.v1.placeholder(tf.float32, 1)

z1 = tf.multiply(x1, w1)
z2 = tf.multiply(x2, w2)

output = tf.add(z1, z2)

with tf.compat.v1.Session() as sess:
    print(sess.run(output, feed_dict={x1: [2], w1: [2], x2: [4], w2: [4]}))

q1 = tf.constant(1)
q2 = tf.constant(2)

z = tf.add(q1, q2)
init = tf.compat.v1.global_variables_initializer()
print(sess.run(init), sess.run(z))
