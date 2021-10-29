'''
tensorflow实现一个简单的神经网络，用神经网络来预测 y = x^2 - 0.5,
网络的结构很简单，输入层和输出层都只有一个神经元，而隐藏层有10个神经元。
'''
#import tensorflow as tf
import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt

tf.disable_v2_behavior()
tf.print(tf.__version__)

'''
定义网络结构
输入的参数:
该层输入，输入数据的大小，输出数据的大小，以及使用的激活函数，激活函数在默认情况下是None，即不适用激活函数
'''
def add_layer(inputs,in_size,out_size,activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    biases = tf.Variable(tf.zeros([1,out_size])+0.1)
    Wx_plus_b = tf.add(tf.matmul(inputs,Weights),biases)
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

#生成训练数据
#这里，我们生成300*1的x，然后增加一点噪声noise，通过y = x^2 - 0.5+noise来生成y
x_data = np.linspace(-1,1,300)[:,np.newaxis]
# print(x_data)
noise = np.random.normal(0,0.05,x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise

#构建网络
#定义输入层-隐藏层-输出层的三层神经网络结构，其中输入层和输出层仅有一个神经元，而隐藏层有10个神经元。同时，我们定义我们的损失是平方损失函数，通过梯度下降法来最小化我们的损失。
#None表示给多少个sample都可以
xs = tf.placeholder(tf.float32,[None,1])
ys = tf.placeholder(tf.float32,[None,1])
# add hidden layer
l1 = add_layer(xs,1,10,activation_function=tf.nn.relu)
# add output layer
prediction = add_layer(l1,10,1,activation_function=None)
#在计算平方损失的时候，我们先使用tf.reduce_sum来计算了每一个样本点的损失，注意这里的参数 reduction_indices=[1]，这表明我们是在第1维上运算
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                     reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

#定义Session并训练
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
        if i % 50 == 0:
            print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))