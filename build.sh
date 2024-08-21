set -e

bazel build -c opt --copt -g --strip=never --copt=-mavx --copt=-mavx2 --copt=-D_GLIBCXX_USE_CXX11_ABI=0 --config=mkl_threadpool --noincompatible_disable_nocopts --host_force_python=PY2 //tensorflow:libtensorflow_cc.so
bazel build -c opt --copt -g --strip=never --copt=-mavx --copt=-mavx2 --copt=-D_GLIBCXX_USE_CXX11_ABI=0 --config=mkl_threadpool --noincompatible_disable_nocopts --host_force_python=PY2 //tensorflow:libtensorflow_framework.so
