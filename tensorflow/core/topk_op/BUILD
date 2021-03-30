load(
    "//tensorflow:tensorflow.bzl",
    "tf_kernel_library",
    "tf_copts",
)

tf_kernel_library(
    name = "gtopk_op",
    srcs = [
      "GTopK_kernel.cc"
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core/kernels:eigen_helpers",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib"
     ],
)
