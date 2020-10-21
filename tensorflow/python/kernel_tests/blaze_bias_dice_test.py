from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python import tf2
from tensorflow.python.client import session
from tensorflow.python.compat import compat
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import benchmark
from tensorflow.python.platform import test


def GetRandomNormalInput(shape, dtype, loc=0.0, scale=1.0):
  vals = np.array(np.random.normal(loc, scale, np.prod(shape)), dtype=dtype)
  if dtype in (np.complex64, np.complex128):
    imag = np.array(np.random.normal(loc, scale, np.prod(shape)), dtype=dtype)
    vals += 1j * imag
  return vals.reshape(shape)


def NPSigmoid(x):
  return 1.0 / (1.0 + np.exp(-x))


class BlazeBiasDiceOpTest(test.TestCase):
  def _npBlazeBiasDice(self, input, bias, alpha, moving_mean, gamma):
    fc_out = input + bias
    bn_out = alpha * (fc_out - moving_mean)
    logits = NPSigmoid(bn_out)
    out = gamma * (1.0 - logits) * fc_out + logits * fc_out
    return out

  def testCPUCorrect(self):
    def Compare(batch, units, dtype):
      input_in = GetRandomNormalInput([batch, units], dtype)
      bias_in = GetRandomNormalInput([units], dtype)
      alpha_in = GetRandomNormalInput([units], dtype)
      moving_mean_in = GetRandomNormalInput([units], dtype)
      gamma_in = GetRandomNormalInput([units], dtype)
      if dtype == "float16":
        tol = 0.01
      elif dtype == "float32":
        tol = 1e-5
      with self.cached_session(use_gpu=False) as sess:
        input_ph = array_ops.placeholder(dtype, input_in.shape)
        bias_ph = array_ops.placeholder(dtype, bias_in.shape)
        alpha_ph = array_ops.placeholder(dtype, alpha_in.shape)
        moving_mean_ph = array_ops.placeholder(dtype, moving_mean_in.shape)
        gamma_ph = array_ops.placeholder(dtype, gamma_in.shape)
        z0 = math_ops.blaze_bias_dice(input_ph, bias_ph, alpha_ph,
                                      moving_mean_ph, gamma_ph)
        z0_val = sess.run(z0,
                          feed_dict={
                              input_ph: input_in,
                              bias_ph: bias_in,
                              alpha_ph: alpha_in,
                              moving_mean_ph: moving_mean_in,
                              gamma_ph: gamma_in,
                          })
      z1 = self._npBlazeBiasDice(input_in, bias_in, alpha_in,
                                 moving_mean_in, gamma_in)
      self.assertAllClose(z0_val, z1, rtol=tol, atol=tol)

    Compare(80, 1024, "float32")
    Compare(80, 1024, "float16")

  def testGPUCorrect(self):
    def Compare(batch, units, dtype):
      input_in = GetRandomNormalInput([batch, units], dtype)
      bias_in = GetRandomNormalInput([units], dtype)
      alpha_in = GetRandomNormalInput([units], dtype)
      moving_mean_in = GetRandomNormalInput([units], dtype)
      gamma_in = GetRandomNormalInput([units], dtype)
      if dtype == "float16":
        tol = 0.01
      elif dtype == "float32":
        tol = 1e-5
      with self.cached_session(use_gpu=True) as sess:
        input_ph = array_ops.placeholder(dtype, input_in.shape)
        bias_ph = array_ops.placeholder(dtype, bias_in.shape)
        alpha_ph = array_ops.placeholder(dtype, alpha_in.shape)
        moving_mean_ph = array_ops.placeholder(dtype, moving_mean_in.shape)
        gamma_ph = array_ops.placeholder(dtype, gamma_in.shape)
        z0 = math_ops.blaze_bias_dice(input_ph, bias_ph, alpha_ph,
                                      moving_mean_ph, gamma_ph)
        z0_val = sess.run(z0,
                          feed_dict={
                              input_ph: input_in,
                              bias_ph: bias_in,
                              alpha_ph: alpha_in,
                              moving_mean_ph: moving_mean_in,
                              gamma_ph: gamma_in,
                          })
      z1 = self._npBlazeBiasDice(input_in, bias_in, alpha_in,
                                 moving_mean_in, gamma_in)
      self.assertAllClose(z0_val, z1, atol=tol, rtol=tol)

    Compare(80, 1024, "float32")
    Compare(80, 1024, "float16")
  
if __name__ == '__main__':
  test.main()