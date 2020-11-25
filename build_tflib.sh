#!/bin/sh
#****************************************************************#
# ScriptName: build.sh
# Author: $yunlong.xyl@alibaba-inc.com
# Create Date: 2019-07-22 19:01
# Modify Author: $SHTERM_REAL_USER@alibaba-inc.com
# Modify Date: 2020-09-18 17:49
# Function: 
#***************************************************************#
set -e
set -u
### compile debug version
#bazel build --config monolithic --copt=-g --strip=never -c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt "-DEIGEN_ENABLE_AVX2" --copt "-DNDEBUG" //tensorflow:tensorflow
bazel build --config monolithic -c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt "-DEIGEN_ENABLE_AVX2" --copt "-DNDEBUG" --copt "-DINTEL_MKL -DINTEL_MKL_ML" --config=mkl //tensorflow:tensorflow
