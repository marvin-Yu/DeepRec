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


def GetRandomNormalInput(shape, dtype):
    # float16 has limited range so we reduce the variance of the scalars.
    scale = 10.0 if dtype != np.float16 else 0.1
    loc = -10.0 if dtype != np.float16 else 0.1
    vals = np.array(np.random.normal(loc, scale, np.prod(shape)), dtype=dtype)
    if dtype in (np.complex64, np.complex128):
        imag = np.array(np.random.normal(loc, scale, np.prod(shape)),
                        dtype=dtype)
        vals += 1j * imag
    return vals.reshape(shape)


class ParallelIndicatorMatMulOpTest(test.TestCase):
    def _npIndicatorMatMul(self, x, y, indicator):
        x = x[:, indicator, :, :]
        return np.matmul(x, y)

    def testCPUCorrect(self):
        def Compare(parallel, batch_a, batch_b, m, n, k):
            x_in = GetRandomNormalInput([parallel, batch_a, m, k], np.float32)
            y_in = GetRandomNormalInput([parallel, batch_b, k, n], np.float32)
            ind_in = np.random.randint(0,
                                       batch_a,
                                       size=batch_b,
                                       dtype=np.int64)
            tol = 1e-4
            with self.cached_session(use_gpu=False) as sess:
                x_ph = array_ops.placeholder(x_in.dtype)
                y_ph = array_ops.placeholder(y_in.dtype)
                ind_ph = array_ops.placeholder(ind_in.dtype)
                z0 = math_ops.parallel_indicator_mat_mul(
                    x_ph, y_ph, ind_ph, parallel)
                z0_val = sess.run(z0,
                                  feed_dict={
                                      x_ph: x_in,
                                      y_ph: y_in,
                                      ind_ph: ind_in
                                  })
            z1 = self._npIndicatorMatMul(x_in, y_in, ind_in)
            self.assertAllClose(z0_val, z1, rtol=tol, atol=tol)

        Compare(14, 25, 200, 400, 4, 5)
        Compare(7, 3, 5, 6, 7, 5)
        Compare(1244, 34, 63, 6, 12, 12)

    def testGPUCorrect(self):
        def Compare(parallel, batch_a, batch_b, m, n, k):
            x_in = GetRandomNormalInput([parallel, batch_a, m, k], np.float32)
            y_in = GetRandomNormalInput([parallel, batch_b, k, n], np.float32)
            ind_in = np.random.randint(0,
                                       batch_a,
                                       size=batch_b,
                                       dtype=np.int64)
            tol = 1e-4
            with self.cached_session(use_gpu=True) as sess:
                x_ph = array_ops.placeholder(x_in.dtype)
                y_ph = array_ops.placeholder(y_in.dtype)
                ind_ph = array_ops.placeholder(ind_in.dtype)
                z0 = math_ops.parallel_indicator_mat_mul(
                    x_ph, y_ph, ind_ph, parallel)
                z0_val = sess.run(z0,
                                  feed_dict={
                                      x_ph: x_in,
                                      y_ph: y_in,
                                      ind_ph: ind_in
                                  })
            z1 = self._npIndicatorMatMul(x_in, y_in, ind_in)
            self.assertAllClose(z0_val, z1, rtol=tol, atol=tol)

        Compare(14, 25, 200, 400, 4, 5)
        Compare(7, 3, 5, 6, 7, 5)
        Compare(1244, 34, 63, 6, 12, 12)


class ParallelIndicatorBatchedSmallMatMulOpTest(test.TestCase):
    def _npIndicatorBatchedSmallMatMul(self, x, y, indicator, use_tanh=False):
        x = x[:, indicator, :, :]
        r = np.matmul(x, y)
        if use_tanh:
            r = np.tanh(r)
        return r

    def testCPUCorrect(self):
        def Compare(parallel, batch_a, batch_b, m, n, k, use_tanh):
            x_in = GetRandomNormalInput([parallel, batch_a, m, k], np.float32)
            y_in = GetRandomNormalInput([parallel, batch_b, k, n], np.float32)
            ind_in = np.random.randint(0,
                                       batch_a,
                                       size=batch_b,
                                       dtype=np.int64)
            tol = 1e-4
            with self.cached_session(use_gpu=False) as sess:
                x_ph = array_ops.placeholder(x_in.dtype)
                y_ph = array_ops.placeholder(y_in.dtype)
                ind_ph = array_ops.placeholder(ind_in.dtype)
                z0 = math_ops.parallel_indicator_batched_small_mat_mul(
                    x_ph, y_ph, ind_ph, parallel, use_tanh=use_tanh)
                z0_val = sess.run(z0,
                                  feed_dict={
                                      x_ph: x_in,
                                      y_ph: y_in,
                                      ind_ph: ind_in
                                  })
            z1 = self._npIndicatorBatchedSmallMatMul(x_in, y_in, ind_in,
                                                     use_tanh)
            self.assertAllClose(z0_val, z1, rtol=tol, atol=tol)

        Compare(14, 25, 200, 400, 4, 5, False)
        Compare(7, 3, 5, 6, 7, 5, False)
        Compare(1244, 34, 63, 6, 12, 12, False)
        Compare(14, 25, 200, 400, 4, 5, True)
        Compare(7, 3, 5, 6, 7, 5, True)
        Compare(1244, 34, 63, 6, 12, 12, True)

    def testCPUCorrect(self):
        def Compare(parallel, batch_a, batch_b, m, n, k, use_tanh):
            x_in = GetRandomNormalInput([parallel, batch_a, m, k], np.float32)
            y_in = GetRandomNormalInput([parallel, batch_b, k, n], np.float32)
            ind_in = np.random.randint(0,
                                       batch_a,
                                       size=batch_b,
                                       dtype=np.int64)
            tol = 1e-4
            with self.cached_session(use_gpu=True) as sess:
                x_ph = array_ops.placeholder(x_in.dtype)
                y_ph = array_ops.placeholder(y_in.dtype)
                ind_ph = array_ops.placeholder(ind_in.dtype)
                z0 = math_ops.parallel_indicator_batched_small_mat_mul(
                    x_ph, y_ph, ind_ph, parallel, use_tanh=use_tanh)
                z0_val = sess.run(z0,
                                  feed_dict={
                                      x_ph: x_in,
                                      y_ph: y_in,
                                      ind_ph: ind_in
                                  })
            z1 = self._npIndicatorBatchedSmallMatMul(x_in, y_in, ind_in,
                                                     use_tanh)
            self.assertAllClose(z0_val, z1, rtol=tol, atol=tol)

        Compare(14, 25, 200, 400, 4, 5, False)
        Compare(7, 3, 5, 6, 7, 5, False)
        Compare(1244, 34, 63, 6, 12, 12, False)
        Compare(14, 25, 200, 400, 4, 5, True)
        Compare(7, 3, 5, 6, 7, 5, True)
        Compare(1244, 34, 63, 6, 12, 12, True)


class ParallelIndicatorMatMulBenchmark(test.Benchmark):
    params = [
        (14, 25, 200, 400, 4, 5),
        (14, 25, 200, 800, 4, 5),
    ]

    def benchmarkParallelIndicatorMatMul(self):
        for p in self.params:
            parallel, batch_a, batch_b, m, n, k = p
            with ops.Graph().as_default(), \
                session.Session(config=benchmark.benchmark_config()) as sess, \
                    ops.device("/gpu:0"):
                x = variables.Variable(
                    GetRandomNormalInput([parallel, batch_a, m, k],
                                         np.float32))
                y = variables.Variable(
                    GetRandomNormalInput([parallel, batch_b, k, n],
                                         np.float32))
                ind = variables.Variable(
                    np.random.randint(0, batch_a, size=batch_b,
                                      dtype=np.int64))
                variables.global_variables_initializer().run()
                self.run_op_benchmark(
                    sess,
                    math_ops.parallel_indicator_mat_mul(x, y, ind, parallel),
                    min_iters=20,
                    name="parallel_indicator_mat_mul_{}_{}".format(
                        x.shape, y.shape))


class ParallelIndicatorBatchedSmallMatMulBenchmark(test.Benchmark):
    params = [
        (14, 25, 200, 400, 4, 5, False),
        (14, 25, 200, 400, 4, 5, True),
        (14, 25, 200, 800, 4, 5, False),
        (14, 25, 200, 800, 4, 5, True),
    ]

    def benchmarkParallelIndicatorBatchedSmallMatMul(self):
        for p in self.params:
            parallel, batch_a, batch_b, m, n, k, use_tanh = p
            with ops.Graph().as_default(), \
                session.Session(config=benchmark.benchmark_config()) as sess, \
                    ops.device("/gpu:0"):
                x = variables.Variable(
                    GetRandomNormalInput([parallel, batch_a, m, k],
                                         np.float32))
                y = variables.Variable(
                    GetRandomNormalInput([parallel, batch_b, k, n],
                                         np.float32))
                ind = variables.Variable(
                    np.random.randint(0, batch_a, size=batch_b,
                                      dtype=np.int64))
                variables.global_variables_initializer().run()
                self.run_op_benchmark(
                    sess,
                    math_ops.parallel_indicator_batched_small_mat_mul(
                        x, y, ind, parallel, use_tanh=use_tanh),
                    min_iters=20,
                    name="parallel_indicator_mat_mul_{}_{}".format(
                        x.shape, y.shape))


if __name__ == '__main__':
    test.main()