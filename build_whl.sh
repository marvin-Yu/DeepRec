#!/bin/sh
#****************************************************************#
# ScriptName: build.sh
# Author: $SHTERM_REAL_USER@alibaba-inc.com
# Create Date: 2020-12-03 17:40
# Modify Author: $SHTERM_REAL_USER@alibaba-inc.com
# Modify Date: 2021-02-07 10:03
# Function:
#***************************************************************#
set -e

bazel build --copt=-mavx2 --copt "-DINTEL_MKL_MATMUL"  -c opt --config=cuda  --copt -mfpmath=both --copt -mfma --copt -msse4.2 --copt -D_GLIBCXX_USE_CXX11_ABI=0  //tensorflow/tools/pip_package:build_pip_package
#bazel build --copt=-mavx2 --copt "-DINTEL_MKL -DINTEL_MKL_ML" --config=mkl -c opt --config=cuda  --copt -mfpmath=both --copt -mfma --copt -msse4.2 --copt -D_GLIBCXX_USE_CXX11_ABI=0  //tensorflow/tools/pip_package:build_pip_package

bazel-bin/tensorflow/tools/pip_package/build_pip_package ~/tensorflow_pkg
