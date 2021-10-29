'''
How to use session in TensorFlow
'''
import tensorflow.compat.v1 as tf

matrix1 = tf.constant([[3, 3]])
matrix2 = tf.constant([[2],
                       [2]])
product = tf.matmul(matrix1, matrix2)

# Method 1
sess = tf.Session()
res  = sess.run(product)
print(res)
sess.close()

# Method 2
with tf.Session() as sess:
    res  = sess.run(product)
    print(res)