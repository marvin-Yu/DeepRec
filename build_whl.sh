#!/bin/sh
#****************************************************************#
# ScriptName: build.sh
# Author: $SHTERM_REAL_USER@alibaba-inc.com
# Create Date: 2020-12-03 17:40
# Modify Author: $SHTERM_REAL_USER@alibaba-inc.com
# Modify Date: 2021-10-16 16:03
# Function: 
#***************************************************************#
set -e 

#bazel build --copt=-mavx2 -c opt --config=cuda --copt -mfpmath=both --copt -mfma --copt -msse4.2 --copt -D_GLIBCXX_USE_CXX11_ABI=0 --copt -DGOOGLE_CUDA=1 //tensorflow/tools/pip_package:build_pip_package 
bazel build --copt=-mavx2 --copt='-DINTEL_MKL_GEMM_ONLY' --config=mkl_gemm_only -c opt --config=cuda  --copt -mfpmath=both --copt -mfma --copt -msse4.2 --copt -DGOOGLE_CUDA=1 --copt -D_GLIBCXX_USE_CXX11_ABI=0 //tensorflow/tools/pip_package:build_pip_package

bazel-bin/tensorflow/tools/pip_package/build_pip_package ~/tensorflow_pkg

echo y| pip uninstall tensorflow

pip install --user ~/tensorflow_pkg/tensorflow-1.15.0rc1-cp37-cp37m-linux_x86_64.whl
