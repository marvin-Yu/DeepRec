load(
    "//tensorflow:tensorflow.bzl",
    "tf_kernel_library",
    "tf_copts",
)

cc_library(
    name = "gru_op_lib",
    srcs = [
        "gru_op.cc",
    ],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:framework",
        ":gru_op",
    ],
    alwayslink = 1,
)

tf_kernel_library(
    name = "gru_op",
    srcs = [
      "gru_func.h",
      "gru_kernel.cc",
      "gru_func.cc"
    ],
    gpu_srcs = [
      "gru_func.h",
      "gru_func.cu.cc"
    ],
    deps = [
        "//tensorflow/core/kernels:eigen_helpers",
    ],
)
