'''
placeholder in TensorFlow
中文翻译为占位符
'''
import tensorflow.compat.v1 as tf

# define a placeholder
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
input3 = tf.placeholder(tf.float32, [2, 2]) # define a shape [2, 2] placeholder
output = tf.multiply(input1, input2) # output = input1 * input2

init = tf.global_variables_initializer()
with tf.Session() as sess:
    # becase input1 and input2 just placeholder, don't have value
    print(sess.run(output, feed_dict={input1: [3.], input2: [6.]}))
