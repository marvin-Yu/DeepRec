import tensorflow as tf

jit_scope = tf.contrib.compiler.jit.experimental_jit_scope
ni = tf.constant(-1.0, shape=[16,288], dtype=tf.float32, name='x_hold')
w1 = tf.constant(-1.0, shape=[288,576], dtype=tf.float32, name='y_hold')
with tf.device('/device:FPGA:0'):
	with jit_scope():
  		o4 = tf.matmul(ni, w1, name='prod1') 

sess = tf.Session()
result = sess.run(o4)
print result

