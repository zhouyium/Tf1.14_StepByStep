'''
本次课程将通过200次学习的方式来逼近w=0.1 b=0.3
'''
import tensorflow.compat.v1 as tf
import numpy as np

# create data
x_data=np.random.rand(100).astype(np.float32)
y_data=x_data*0.1 + 0.3

# create TensorFlow structure
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases  = tf.Variable(tf.zeros([1]))
y = Weights * x_data + biases

# define loss
loss = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# define TensorFlow session
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# training
for step in range(201):
    sess.run(train)
    if 0==(step % 20):
        print(step, sess.run(Weights), sess.run(biases))