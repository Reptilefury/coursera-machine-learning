import tensorflow as tf



constant_x = tf.constant(5, name='constant_x')
variable_y = tf.Variable(constant_x + 5, name='variable_y')
print(variable_y)
init = tf.compat.v1.global_variables_initializer()

print(variable_y)
