import tensorflow as tf

jit_scope = tf.contrib.compiler.jit.experimental_jit_scope

ni = tf.random_uniform([256,288],minval=-2,maxval=3,dtype=tf.float32)

w1 = tf.Variable(tf.random_uniform([288, 576],minval=-2,maxval=3,dtype=tf.float32))
b1 = tf.Variable(tf.random_uniform([1, 576],minval=-2,maxval=3,dtype=tf.float32))

w2 = tf.Variable(tf.random_uniform([576, 288],minval=-2,maxval=3,dtype=tf.float32))
b2 = tf.Variable(tf.random_uniform([1, 288],minval=-2,maxval=3,dtype=tf.float32))

w3 = tf.Variable(tf.random_uniform([288, 192],minval=-2,maxval=3,dtype=tf.float32))
b3 = tf.Variable(tf.random_uniform([1, 192],minval=-2,maxval=3,dtype=tf.float32))

w4 = tf.Variable(tf.random_uniform([192, 96],minval=-2,maxval=3,dtype=tf.float32))
b4 = tf.Variable(tf.random_uniform([1, 96],minval=-2,maxval=3,dtype=tf.float32))

with tf.device('/device:FPGA:0'):
	with jit_scope():
  		o1 = tf.nn.relu(tf.matmul(ni, w1, name='prod1') + b1)
  		o2 = tf.nn.relu(tf.matmul(o1, w2, name='prod2') + b2)
  		o3 = tf.nn.relu(tf.matmul(o2, w3, name='prod3') + b3)
        	o4 =            tf.matmul(o3, w4, name='prod4') + b4

sess = tf.Session()
sess.run(tf.global_variables_initializer())
result = sess.run(o4)

print result

