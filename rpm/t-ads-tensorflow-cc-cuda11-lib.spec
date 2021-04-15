##############################################################
# http://twiki.corp.alimama.com/twiki/bin/view/Alimm_OPS/RPM #
# http://www.rpm.org/max-rpm/ch-rpm-inside.html              #
##############################################################
Name: %(echo t-ads-tensorflow-cc-cuda11-lib${SUFFIX})
Packager:xianjie.qxj
Version:1.15.0
# if you want get version number from outside, use like this
Release:%(echo $RELEASE)%{?dist}

%global _enable_debug_package 0
%global debug_package %{nil}
%global __os_install_post /usr/lib/rpm/brp-compress %{nil}

# if you want use the parameter of rpm_create on build time,
# uncomment below
Summary:tensorflow cc library

URL: http://gitlab.alibaba-inc.com/TargetAdvertising/tensorflow
Group: alimama
License: Commercial
BuildRoot: %{BUILD}/%{name}-%{version}-%{release}
BuildArch: x86_64
AutoReq: no

%description
CodeUrl:git@gitlab.alibaba-inc.com:TargetAdvertising/tensorflow.git blaze/master
# if you want publish current git URL or Revision use these macros
%{_git_path}
%{_git_revision}
Alimama alogserver for display ads

%prep

%build

# down load bazel cache file from oss
rm -rf /home/admin/.cache/bazel/_bazel_admin/
wget -q http://211619.oss-cn-hangzhou-zmf.aliyuncs.com/public/tf_115_cache.tgz\
  && mkdir -p /home/admin/.cache/bazel/_bazel_admin/ \
  && tar -zxvf tf_115_cache.tgz -C /home/admin/.cache/bazel/_bazel_admin/\
  && rm -rf tf_115_cache.tgz

WORK_DIR=$OLDPWD/../
cd $WORK_DIR
export TEST_TMPDIR=/home/admin/.cache/bazel/
env PYTHON_BIN_PATH=/opt/conda/bin/python \
    PYTHON_LIB_PATH="/opt/conda/lib/python3.7/site-packages" \
    TF_ENABLE_XLA=1 TF_NEED_OPENCL_SYCL=0 TF_NEED_ROCM=0 \
    TF_NEED_CUDA=1 TF_NEED_TENSORRT=0 TF_CUDA_CLANG=0 \
    GCC_HOST_COMPILER_PATH=/usr/bin/gcc TF_NEED_MPI=0 \
    CC_OPT_FLAGS="-march=native -Wno-sign-compare" \
    TF_CUDA_VERSION=11.2 \
    CUDA_TOOLKIT_PATH=/usr/local/cuda-11.2 \
    TF_CUDNN_VERSION=8.1.1 \
    TF_CUDA_COMPUTE_CAPABILITIES="6.0,7.0,7.5,8.0,8.6" \
    LD_LIBRARY_PATH="/usr/local/cuda-11.2/lib64:/usr/local/cuda-11.2/extras/CUPTI/lib64/:" \
    TF_SET_ANDROID_WORKSPACE=0 ./configure
export LD_LIBRARY_PATH="/usr/local/cuda-11.2/lib64:/usr/local/cuda-11.2/extras/CUPTI/lib64/:"
sh install.sh

%install
export DONT_STRIP=1

mkdir -p .%{_prefix}/tensorflow/include
mkdir -p .%{_prefix}/tensorflow/lib

cp -r $OLDPWD/../../_external/usr/local/include/* .%{_prefix}/tensorflow/include
cp -a $OLDPWD/../../_external/usr/local/lib64/* .%{_prefix}/tensorflow/lib

%files
%defattr(-,ads,users)
%{_prefix}

%post
echo "Now ldconfig..."
/sbin/ldconfig

%changelog
