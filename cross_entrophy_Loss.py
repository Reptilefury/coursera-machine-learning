import tensorflow as tf
tf.compat.v1.disable_eager_execution()

softmax_data = [0.1, 0.5, 0.4]
onehot_data = [0.0, 1.0, 0.0]
softmax = tf.compat.v1.placeholder(tf.float32)
onehot_encoding = tf.compat.v1.placeholder(tf.float32)

cross_entropy = tf.reduce_sum(tf.multiply(onehot_encoding, tf.compat.v1.log(softmax)))
cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(logits=tf.compat.v1.log(softmax), labels=onehot_encoding)
with tf.compat.v1.Session() as sess:
    print(sess.run(cross_entropy, feed_dict={softmax: softmax_data, onehot_encoding: onehot_data}))
    print(sess.run(cross_entropy_loss, feed_dict={softmax: softmax_data, onehot_encoding: onehot_data}))
