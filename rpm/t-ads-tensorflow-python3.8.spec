##############################################################
# http://twiki.corp.alimama.com/twiki/bin/view/Alimm_OPS/RPM #
# http://www.rpm.org/max-rpm/ch-rpm-inside.html              #
##############################################################
Name: %(echo t-ads-tensorflow-python3.8${SUFFIX})
Packager:jinluyang.jly
Version:1.15.0
Requires: python
# if you want get version number from outside, use like this
Release:%(echo $RELEASE)%{?dist}

%global _enable_debug_package 0
%global debug_package %{nil}
%global __os_install_post /usr/lib/rpm/brp-compress %{nil}

# if you want use the parameter of rpm_create on build time,
# uncomment below
Summary:tensorflow python3.8 wheel

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
wget http://211619.oss-cn-hangzhou-zmf.aliyuncs.com/public/tf_115_cache.tgz\
  && mkdir -p /home/admin/.cache/bazel/_bazel_admin/ \
  && tar -zxvf tf_115_cache.tgz -C /home/admin/.cache/bazel/_bazel_admin/\
  && rm -rf tf_115_cache.tgz

WORK_DIR=$OLDPWD/../
cd $WORK_DIR
export TEST_TMPDIR=/home/admin/.cache/bazel/
env PYTHON_BIN_PATH=/usr/local/python3/bin/python \
    PYTHON_LIB_PATH="/usr/local/python3/lib/python3.8/site-packages" \
    TF_ENABLE_XLA=1 TF_NEED_OPENCL_SYCL=0 TF_NEED_ROCM=0 \
    TF_NEED_CUDA=1 TF_NEED_TENSORRT=0 TF_CUDA_CLANG=0 \
    GCC_HOST_COMPILER_PATH=/usr/bin/gcc TF_NEED_MPI=0 \
    CC_OPT_FLAGS="-march=native -Wno-sign-compare" \
    CUDA_TOOLKIT_PATH=/usr/local/cuda-10.1 \
    TF_CUDA_COMPUTE_CAPABILITIES="6.0,7.0,7.5" \
    LD_LIBRARY_PATH="/usr/local/cuda-10.1/lib64:/usr/local/cuda-10.1/extras/CUPTI/lib64/:" \
    TF_SET_ANDROID_WORKSPACE=0 ./configure
export LD_LIBRARY_PATH="/usr/local/cuda-10.1/lib64:/usr/local/cuda-10.1/extras/CUPTI/lib64/:"
sh build_pip.sh

%install
cp -r /tmp/tensorflow_pkg/* .%{_prefix}/


%files
%defattr(-,ads,users)
%{_prefix}

%post
echo "Now ldconfig..."
/sbin/ldconfig
#it cannot find pip
#echo y| pip uninstall tensorflow
#pip install --user %{_prefix}/tensorflow-1.15.0rc1-cp38-cp38-linux_x86_64.whl
echo "numpy need to be <= 1.18.0 when using bfloat because numpy ABI has changed"
echo ".whl is at /home/a , use below command to pip install"
echo "echo y| pip uninstall tensorflow"
echo "pip install --user tensorflow-1.15.0*.whl"

%changelog
