#export TEST_TMPDIR=
#python ./configure.py
declare -a targets=("//tensorflow:libtensorflow_framework.so"
                    "//tensorflow:libtensorflow_cc.so")
declare -a install_targets=("libtensorflow_framework.so.1.15.0"
                            "libtensorflow_framework.so.1"
                            "libtensorflow_framework.so"
                            "libtensorflow_cc.so.1.15.0"
                            "libtensorflow_cc.so.1"
                            "libtensorflow_cc.so")
# now loop through the above array
for target in "${targets[@]}"
do
    bazel build --copt=-mavx2 -c opt --copt -g --config=cuda --copt -D_GLIBCXX_USE_CXX11_ABI=0 $target
done

EXTERNAL_DIR="../_external"
EXTERNAL_DIR=`readlink -f $EXTERNAL_DIR`
HEADER_DIR=$EXTERNAL_DIR"/usr/local/include/"

CURRENT_DIR=`basename $PWD`
BAZEL_CACHE_DIR=`readlink bazel-$CURRENT_DIR`/../../
BAZEL_EXTERNAL_DIR=$BAZEL_CACHE_DIR"/external/"

for target in "${install_targets[@]}"
do
    IFS='/' read -ra path <<< "$target"
    rm $EXTERNAL_DIR/usr/local/lib64/${path[-1]} -f
    cp -d bazel-bin/tensorflow/$target $EXTERNAL_DIR/usr/local/lib64/
done

# copy header
rm $HEADER_DIR/tensorflow -rf
mkdir -p $HEADER_DIR/tensorflow
find tensorflow/core -name '*.h' -exec cp --parents \{\} $HEADER_DIR/ \;
find tensorflow/core -name '*.proto' -exec cp --parents \{\} $HEADER_DIR/ \;
find tensorflow/c -name '*.h' -exec cp --parents \{\} $HEADER_DIR/ \;
find tensorflow/cc -name '*.h' -exec cp --parents \{\} $HEADER_DIR/ \;
if [ -d bazel-genfiles ]; then
   cd bazel-genfiles
   find tensorflow/ -name '*.h' -exec cp --parents \{\} $HEADER_DIR/ \;
   cd -
fi
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

DITING_DIR=$BAZEL_EXTERNAL_DIR"diting_repo/sdk/include/diting/"
rm $HEADER_DIR/diting -rf
cp -r $DITING_DIR $HEADER_DIR/diting


FARMHASH=$BAZEL_EXTERNAL_DIR"farmhash_archive/src/farmhash.h"
cp $FARMHASH $HEADER_DIR
