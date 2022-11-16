# Description:
# Python APIs for various Tensorflow backends.

load("//tensorflow:tensorflow.bzl", "if_not_windows")
load(
    "@local_config_tensorrt//:build_defs.bzl",
    "if_tensorrt_v6",
    "if_tensorrt_v8"
)

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["LICENSE"])

py_library(
    name = "compiler",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = if_tensorrt_v6([
        "//tensorflow/python/compiler/tensorrt:init_py",
    ]) + if_tensorrt_v8([
        "//tensorflow/python/compiler/tensorrt8:init_py",
    ]) + [
        "//tensorflow/python/compiler/xla:compiler_py",
    ],
)
