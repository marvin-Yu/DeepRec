load(
    "//tensorflow:tensorflow.bzl",
    "tf_kernel_library",
    "tf_copts",
)

tf_kernel_library(
    name = "gru_op",
    srcs = [
      "gru_func.h",
      "gru_kernel.cc",
      "gru_func.cc",
      "vml.h",
      "vml.cc"
    ],
    gpu_srcs = [
      "gru_func.h",
      "gru_func.cu.cc"
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
