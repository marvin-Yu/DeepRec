#!/bin/sh
#****************************************************************#
# ScriptName: build.sh
# Author: $yunlong.xyl@alibaba-inc.com
# Create Date: 2019-07-22 19:01
# Modify Author: $SHTERM_REAL_USER@alibaba-inc.com
# Modify Date: 2019-08-13 16:45
# Function: 
#***************************************************************#
set -e
set -u
### compile debug version
#bazel build --config monolithic --copt=-g --strip=never -c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt "-DEIGEN_ENABLE_AVX2" --copt "-DNDEBUG" //tensorflow:tensorflow
bazel build --config monolithic -c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt "-DEIGEN_ENABLE_AVX2" --copt "-DNDEBUG" //tensorflow:tensorflow
