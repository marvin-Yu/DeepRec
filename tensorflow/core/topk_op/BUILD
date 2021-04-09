load(
    "//tensorflow:tensorflow.bzl",
    "tf_kernel_library",
    "tf_copts",
)

tf_kernel_library(
    name = "gtopk_op",
    srcs = [
      "util.h",
      "GTopK_kernel.cc",
      "IdxGTopK_kernel.cc",
    ],
    visibility = ["//visibility:public"],
)
