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


def NPSoftmax(logits):
  shift_logits = logits - np.max(logits, axis=-1, keepdims=True)
  t = np.exp(shift_logits)
  return t / np.sum(t, axis=-1, keepdims=True)


class BlazeAttentionOpTest(test.TestCase):
  def _npBlazeAttention(self, fact, query):
    pnum, _, seq, units = fact.shape
    _, bs, _ = query.shape
    query = np.reshape(query, [pnum, bs, 1, units])
    logits = np.sum(fact * query, axis=3)
    score = np.reshape(NPSoftmax(logits), [pnum, bs, seq, 1])
    r = np.sum(score * fact, axis=2)
    return np.transpose(r, (1, 0, 2))

  def testCPUCorrect(self):
    def Compare(pnum, bs, seq, units):
      fact_in = GetRandomNormalInput([pnum, 1, seq, units], np.float32)
      query_in = GetRandomNormalInput([pnum, bs, units], np.float32)
      tol = 1e-5
      with self.cached_session(use_gpu=False) as sess:
        fact_ph = array_ops.placeholder(fact_in.dtype, fact_in.shape)
        query_ph = array_ops.placeholder(query_in.dtype, query_in.shape)
        z0 = math_ops.blaze_attention(fact_ph, query_ph)
        z0_val = sess.run(z0, feed_dict={fact_ph: fact_in, query_ph: query_in})
      z1 = self._npBlazeAttention(fact_in, query_in)
      self.assertAllClose(z0_val, z1, rtol=tol, atol=tol)

    Compare(12, 200, 150, 32)
    Compare(3, 80, 50, 32)
    Compare(3, 80, 200, 32)
    Compare(3, 100, 100, 32)

  def testGPUCorrect(self):
    def Compare(pnum, bs, seq, units):
      fact_in = GetRandomNormalInput([pnum, 1, seq, units], np.float32)
      query_in = GetRandomNormalInput([pnum, bs, units], np.float32)
      tol = 1e-5
      with self.cached_session(use_gpu=True) as sess:
        fact_ph = array_ops.placeholder(fact_in.dtype, fact_in.shape)
        query_ph = array_ops.placeholder(query_in.dtype, query_in.shape)
        z0 = math_ops.blaze_attention(fact_ph, query_ph)
        z0_val = sess.run(z0, feed_dict={fact_ph: fact_in, query_ph: query_in})
      z1 = self._npBlazeAttention(fact_in, query_in)
      self.assertAllClose(z0_val, z1, rtol=tol, atol=tol)

    Compare(12, 200, 150, 32)
    Compare(3, 80, 50, 32)
    Compare(3, 80, 200, 32)
    Compare(3, 100, 100, 32)

class BlazeAttentionIndicatorOpTest(test.TestCase):
  def _npBlazeAttentionIndicator(self, fact, query, indicators):
    pnum, _, seq, units = fact.shape
    _, bs, _ = query.shape
    fact = fact[:, indicators, :, :]
    query = np.reshape(query, [pnum, bs, -1, units])
    logits = np.sum(fact * query, axis=3)
    score = np.reshape(NPSoftmax(logits), [pnum, bs, seq, 1])
    r = np.sum(score * fact, axis=2)
    return np.transpose(r, (1, 0, 2))

  def testCPUCorrect(self):
    def Compare(pnum, bh, bs, seq, units):
      fact_in = GetRandomNormalInput([pnum, bh, seq, units], np.float32)
      query_in = GetRandomNormalInput([pnum, bs, units], np.float32)
      ind_in = np.random.randint(0, bh, size=bs, dtype=np.int64)
      tol = 1e-5
      with self.cached_session(use_gpu=False) as sess:
        fact_ph = array_ops.placeholder(fact_in.dtype, fact_in.shape)
        query_ph = array_ops.placeholder(query_in.dtype, query_in.shape)
        ind_ph = array_ops.placeholder(ind_in.dtype, ind_in.shape)
        z0 = math_ops.blaze_attention_indicator(fact_ph, query_ph, ind_ph)
        z0_val = sess.run(z0,
                          feed_dict={
                              fact_ph: fact_in,
                              query_ph: query_in,
                              ind_ph: ind_in
                          })
      z1 = self._npBlazeAttentionIndicator(fact_in, query_in, ind_in)
      self.assertAllClose(z0_val, z1, rtol=tol, atol=tol)

    Compare(12, 25, 200, 150, 32)
    Compare(3, 25, 80, 50, 32)
    Compare(3, 12, 80, 200, 32)
    Compare(3, 12, 100, 100, 32)

  def testGPUCorrect(self):
    def Compare(pnum, bh, bs, seq, units):
      fact_in = GetRandomNormalInput([pnum, bh, seq, units], np.float32)
      query_in = GetRandomNormalInput([pnum, bs, units], np.float32)
      ind_in = np.random.randint(0, bh, size=bs, dtype=np.int64)
      tol = 1e-5
      with self.cached_session(use_gpu=True) as sess:
        fact_ph = array_ops.placeholder(fact_in.dtype, fact_in.shape)
        query_ph = array_ops.placeholder(query_in.dtype, query_in.shape)
        ind_ph = array_ops.placeholder(ind_in.dtype, ind_in.shape)
        z0 = math_ops.blaze_attention_indicator(fact_ph, query_ph, ind_ph)
        z0_val = sess.run(z0,
                          feed_dict={
                              fact_ph: fact_in,
                              query_ph: query_in,
                              ind_ph: ind_in
                          })
      z1 = self._npBlazeAttentionIndicator(fact_in, query_in, ind_in)
      self.assertAllClose(z0_val, z1, rtol=tol, atol=tol)

    Compare(12, 25, 200, 150, 32)
    Compare(3, 25, 80, 50, 32)
    Compare(3, 12, 80, 200, 32)
    Compare(3, 12, 100, 100, 32)


if __name__ == '__main__':
  test.main()
