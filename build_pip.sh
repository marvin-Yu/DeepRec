#bazel build //tensorflow/tools/pip_package:build_pip_package
bazel build --copt=-mavx2 --copt='-DINTEL_MKL_GEMM_ONLY' --config=mkl_gemm_only -c opt --config=cuda  --copt -mfpmath=both --copt -mfma --copt -msse4.2 --copt -DGOOGLE_CUDA=1 --copt -D_GLIBCXX_USE_CXX11_ABI=0 //tensorflow/tools/pip_package:build_pip_package
#bazel build --copt=-mavx2 --copt='-DINTEL_MKL_GEMM_ONLY' --config=mkl_gemm_only -c opt --config=cuda --copt -fexceptions --copt -mfpmath=both --copt -mfma --copt -msse4.2 --copt -D_GLIBCXX_USE_CXX11_ABI=0 //tensorflow/tools/pip_package:build_pip_package
if [ $? -eq 0 ]; then
  ./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
fi
