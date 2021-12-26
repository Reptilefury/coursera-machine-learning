import tensorflow as tf

constant_x = tf.constant(5, name='constant_x')
variable_y = tf.Variable(constant_x + 5, name='variable_y')
# print(variable_y)
init = tf.compat.v1.global_variables_initializer()

# print(variable_y)

x_add = tf.add(1, 2, name='x_add')
y_subtract = tf.subtract(3, 4, name='y_subtract')
z_multiply = tf.multiply(5, 6, name='z_multiply')

print(x_add)
tf.compat.v1.disable_eager_execution()
x_new = tf.compat.v1.placeholder(tf.string)
y_test = tf.compat.v1.placeholder(tf.int32)
x_test = tf.compat.v1.placeholder(tf.float32)

print(x_new, {x_new: 'Testing', y_test: 12, x_test: 12.44})

x_array = tf.compat.v1.placeholder("float", [None, 3])
y_array = x_array * 2

input_data = [[1, 2, 3],
              [4, 5, 6]]

logit_data = [2.0, 1.0, 0.1]
logits = tf.compat.v1.placeholder(tf.float32)
softmax = tf.nn.softmax(logits)

with tf.compat.v1.Session() as sess:
    new_output = sess.run(softmax, feed_dict={logits: logit_data})
    print(new_output)
output4 = sess.run(y_array, feed_dict={x_array: input_data})
print(output4)

n_features = 5
n_labels = 2
weights = tf.compat.v1.truncated_normal((n_features, n_labels))
print(sess.run(weights))
x = tf.constant([[1, 1, 1], [1, 1, 1]])
with tf.compat.v1.Session() as sess:
    print(sess.run(tf.reduce_sum(1, 2, 3)))
    print(sess.run(tf.reduce_sum(x, 0)))
