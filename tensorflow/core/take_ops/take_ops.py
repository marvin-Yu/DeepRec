import tensorflow as tf
from tensorflow.python.framework import ops

#take_module = tf.load_op_library('/home/huimin.yhm/tf/bazel-bin/tensorflow/core/take_ops/take.so')
take_module = tf.load_op_library('/home/huimin.yhm/TensorFlowRS/tensorflow-core/bazel-bin/tensorflow/core/take_ops/take.so')

@ops.RegisterGradient("Take")
def _inner_take_grad_cc(op, grad):
    """
    The gradient for `take` using the operation implemented in C++.

    param op: `take` `Operation` that we are differentiating, which we can use
        to find the inputs and outputs of the original op.
    param grad: gradient with respect to the output of the `take` op.
    return: gradients with respect to the input of `take`.
    """
    N = len(op.inputs)/2
    coords = op.inputs[N:len(op.inputs)-1]
    print "coords", coords
    with tf.device('/device:GPU:0'):
        value_grads = take_module.take_grad(grad, coords)
        assert len(value_grads) == N
        return value_grads+[None]*(N+1)

@ops.RegisterGradient("TakeAxis")
def _inner_take_axis_grad_cc(op, grad):
    """
    The gradient for `take_axis` using the operation implemented in C++.

    param op: `take_axis` `Operation` that we are differentiating, which we can use
        to find the inputs and outputs of the original op.
    param grad: gradient with respect to the output of the `take_axis` op.
    return: gradients with respect to the input of `take_axis`.
    """
    value = op.inputs[0]
    begin = op.inputs[1]

    size = op.get_attr('size')
    axis = op.get_attr('axis')
    reverse = op.get_attr('reverse')
    print size, axis, reverse

    print "input_shape", value.shape
    input_shape = value.shape.as_list()
    axis_size = input_shape[axis]
    print input_shape, axis_size

    #with tf.device('/device:GPU:0'):
    in_grad = take_module.take_axis(grad, begin, size=axis_size, axis=axis, reverse=not reverse)
    #return [tf.ones(tf.shape(value)), None]
    return [in_grad, None]

def take(**kwargs):
    return take_module.take(**kwargs)

def take_grad(**kwargs):
    return take_module.take_grad(**kwargs)

def take_axis(**kwargs):
    return take_module.take_axis(**kwargs)
