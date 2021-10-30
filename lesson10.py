'''
Softmax Regressions
'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#from tensorflow.examples.tutorials.mnist import input_data
#import tensorflow.compat.v1 as tf
#from tensorflow.compat.v1.keras.datasets.mnist import mnist

import numpy as np
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("e:/zhouyi/MNIST_data", one_hot=True)

'''
定义网络结构
输入的参数:
该层输入，输入数据的大小，输出数据的大小，以及使用的激活函数，激活函数在默认情况下是None，即不适用激活函数
'''
def add_layer(inputs,in_size,out_size, n_layer, activation_function=None):
    # add layer name
    layer_name = "layer%s" % n_layer
    with tf.name_scope("layer"):
        with tf.name_scope("weights"):
            Weights = tf.Variable(tf.random_normal([in_size,out_size]), name="Weight")
            tf.summary.histogram(layer_name+"/Weights", Weights)
        with tf.name_scope("biase"):
            biases = tf.Variable(tf.zeros([1,out_size])+0.1, name="biase")
            tf.summary.histogram(layer_name+"/biases", biases)
        with tf.name_scope("Wx_plus_b"):
            Wx_plus_b = tf.add(tf.matmul(inputs,Weights),biases)
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        tf.summary.histogram(layer_name+"/outputs", outputs)
        return outputs

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result

# define placeholder for intputs to network
xs = tf.placeholder(tf.float32, [None, 784]) # 28*28=784
ys = tf.placeholder(tf.float32, [None, 10])  # 0~9

# add output layer
prediction = add_layer(xs, 784, 10, 1, activation_function=tf.nn.softmax)

# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#定义Session并训练
init = tf.global_variables_initializer()
with tf.Session() as sess:
    merged = tf.summary.merge_all()
    # you can run "tensorboard --logdir=E:\zhouyi\graph" in Win10
    writer = tf.summary.FileWriter("E:\zhouyi\graph", sess.graph)

    sess.run(init)
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys})
        if i % 50 == 0:
            print(compute_accuracy(mnist.test.images, mnist.test.labels))
            result = sess.run(merged, feed_dict={xs:batch_xs,ys:batch_ys})
            writer.add_summary(result, i)