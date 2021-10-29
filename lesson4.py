'''
How to define variable and constant in TensorFlow
'''
import tensorflow.compat.v1 as tf

# define a variable
state = tf.Variable(0, name='counter') # define a variable name is counter
print(state.name)

# define a constant
one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state)) # print the answer