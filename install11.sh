export TEST_TMPDIR=
EXTERNAL_DIR=$(dirname `readlink -f $0`)/../_external
#prepare cuda-11.2
if [ ! -d $EXTERNAL_DIR/usr/local/cuda-11.2  ]; then
    rm -rf cuda-11.2*
    wget http://109717.oss-cn-hangzhou-zmf.aliyuncs.com/cuda-11.2.tar -O cuda-11.2.tar
    tar -xvf cuda-11.2.tar
    cp -r -d cuda-11.2 $EXTERNAL_DIR/usr/local/
fi
#python ./configure.py
export TF_CUDA_PATHS=$EXTERNAL_DIR/usr/local/cuda-11.2/
export CUDA_TOOLKIT_PATH=$EXTERNAL_DIR/usr/local/cuda-11.2/
export TF_CUDA_VERSION=11.2
export TF_CUDALIB_VERSION=11
export LD_LIBRARY_PATH=$EXTERNAL_DIR/usr/local/cuda-11.2/lib64:$EXTERNAL_DIR/usr/local/cuda-11.2/lib64/stubs:$EXTERNAL_DIR/usr/local/cuda-11.2/extras/CUPTI/lib64:$EXTERNAL_DIR/usr/local/cuda-11.2/targets/x86_64-linux/lib/:$LD_LIBRARY_PATH


export CUDNN_INSTALL_PATH=$EXTERNAL_DIR/usr/local/cuda-11.2/
export TF_CUDNN_VERSION=8.1.1

export NCCL_INSTALL_PATH=$EXTERNAL_DIR/usr/local/cuda-11.2/
#export TF_NCCL_VERSION=2.3.7

export TF_CUDA_CLANG=0
export TF_CUDA_COMPUTE_CAPABILITIES="6.0,6.1,7.0,7.5,8.0,8.6"
export TF_NEED_CUDA=1
export TF_ENABLE_XLA=1
export TF_NEED_OPENCL=0
export TF_NEED_OPENCL_SYCL=0
export TF_NEED_ROCM=0
export TF_NEED_TENSORRT=0
export GCC_HOST_COMPILER_PATH=/usr/bin/gcc
export TF_NEED_MPI=0
export CC_OPT_FLAGS="-march=native"
export TF_SET_ANDROID_WORKSPACE=0

declare -a targets=("//tensorflow:libtensorflow_framework.so"
                    "//tensorflow:libtensorflow_cc.so"
                    "//tensorflow/core:test"
                    "//tensorflow/core:testlib"
                    "//tensorflow/core/kernels:ops_testutil"
)
declare -a install_targets=("tensorflow/libtensorflow_framework.so"
                            "tensorflow/libtensorflow_framework.so.1"
                            "tensorflow/libtensorflow_cc.so.1"
                            "tensorflow/libtensorflow_cc.so"
                            "tensorflow/core/libtest.so"
                            "tensorflow/core/libtestlib.so"
                            "tensorflow/core/kernels/libops_testutil.so"
)

if [ -f ".tf_configure.bazelrc.cuda11" ]; then
    cp .tf_configure.bazelrc.cuda11 .tf_configure.bazelrc
else
    python ./configure.py
fi
## now loop through the above array
export TEST_TMPDIR=~/.cache/bazel/_bazel_${USER}/cuda11_cache
for target in "${targets[@]}"
do
    bazel build -c opt --copt -g --copt=-mavx2 --config=cuda --copt -mfpmath=both --copt -mfma --copt -msse4.2 --copt -D_GLIBCXX_USE_CXX11_ABI=0 --copt -DGOOGLE_CUDA=1  $target
done

EXTERNAL_DIR="../_external"
EXTERNAL_DIR=`readlink -f $EXTERNAL_DIR`
HEADER_DIR=$EXTERNAL_DIR"/usr/local/include/tf_cuda11"

CURRENT_DIR=`basename $PWD`
BAZEL_CACHE_DIR=`readlink bazel-$CURRENT_DIR`/../../
BAZEL_EXTERNAL_DIR=$BAZEL_CACHE_DIR"/external/"

mkdir -p $EXTERNAL_DIR/usr/local/lib64/tf_cuda11
for target in "${install_targets[@]}"
do
    IFS='/' read -ra path <<< "$target"
    rm $EXTERNAL_DIR/usr/local/lib64/tf_cuda11/${path[-1]} -f
    cp -f bazel-bin/$target $EXTERNAL_DIR/usr/local/lib64/tf_cuda11/
done

# copy header
rm $HEADER_DIR/tensorflow -rf
mkdir -p $HEADER_DIR/tensorflow
find tensorflow/core -name '*.h' -exec cp --parents \{\} $HEADER_DIR/ \;
find tensorflow/core -name '*.proto' -exec cp --parents \{\} $HEADER_DIR/ \;
find tensorflow/c -name '*.h' -exec cp --parents \{\} $HEADER_DIR/ \;
find tensorflow/cc -name '*.h' -exec cp --parents \{\} $HEADER_DIR/ \;
if [ -d bazel-out/local-opt ]; then
    cd bazel-out/local-opt/genfiles
    find tensorflow/ -name '*.h' -exec cp --parents \{\} $HEADER_DIR/ \;
    cd -
fi
if [ -d bazel-out/local_linux-opt ]; then
   cd bazel-out/local_linux-opt/genfiles
   find tensorflow/ -name '*.h' -exec cp --parents \{\} $HEADER_DIR/ \;
   cd -
fi
if [ -d bazel-out/k8-opt ]; then
   cd bazel-out/k8-opt/genfiles
   find tensorflow/ -name '*.h' -exec cp --parents \{\} $HEADER_DIR/ \;
   cd -
fi

EIGEN_DIR=$BAZEL_EXTERNAL_DIR"eigen_archive"
rm $HEADER_DIR/eigen3 -rf
cp -r $EIGEN_DIR $HEADER_DIR/eigen3
mkdir -p $HEADER_DIR/third_party/
cp -r third_party/eigen3/ $HEADER_DIR/third_party/

NSYNC_DIR=$BAZEL_EXTERNAL_DIR"nsync/public/"
cd $NSYNC_DIR
find . -name '*.h' -exec cp --parents \{\} $HEADER_DIR/ \;
cd -

ABSL_DIR=$BAZEL_EXTERNAL_DIR"com_google_absl/absl/"
rm $HEADER_DIR/absl -rf
cp -r $ABSL_DIR $HEADER_DIR/absl

#DITING_DIR=$BAZEL_EXTERNAL_DIR"diting_repo/sdk/include/diting/"
#rm $HEADER_DIR/diting -rf
#cp -r $DITING_DIR $HEADER_DIR/diting

cp .tf_configure.bazelrc .tf_configure.bazelrc.cuda11
FARMHASH=$BAZEL_EXTERNAL_DIR"farmhash_archive/src/farmhash.h"
cp $FARMHASH $HEADER_DIR
rm -rf cuda-11.2*
