load(
    "//tensorflow:tensorflow.bzl",
    "tf_kernel_library",
    "tf_copts",
)

tf_kernel_library(
    name = "tree_op",
    srcs = [
      "tree_kernel.cc",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core/kernels:eigen_helpers",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/strings",
     ],
)
