import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

jit_scope = tf.contrib.compiler.jit.experimental_jit_scope



c = tf.Variable(tf.random_uniform([288, 288],minval=2,maxval=2,dtype=tf.float32))
h = tf.Variable(tf.random_uniform([288, 288],minval=2,maxval=2,dtype=tf.float32))
x = tf.Variable(tf.random_uniform([288, 288],minval=2,maxval=2,dtype=tf.float32))
w = tf.Variable(tf.random_uniform([576, 288 * 4],minval=2,maxval=2,dtype=tf.float32))
b = tf.Variable(tf.random_uniform([288 * 4],minval=2,maxval=2,dtype=tf.float32))

sigmoid = math_ops.sigmoid
tanh = math_ops.tanh
with tf.device('/device:FPGA:0'):
  with jit_scope():
    inputs = tf.nn.bias_add(tf.matmul(array_ops.concat([x, h], 1), w) , b);
    i, j, f, o = array_ops.split(value=inputs, num_or_size_splits=4, axis=1)
    new_c = (c * sigmoid(f + 0.5) + sigmoid(i) * tanh(j))
    new_h = sigmoid(o) * tanh(new_c) 
    new_tuple = (new_c, new_h)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
result = sess.run(new_tuple)
print result

