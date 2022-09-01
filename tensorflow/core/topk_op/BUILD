load(
    "//tensorflow:tensorflow.bzl",
    "tf_kernel_library",
    "tf_copts",
)

tf_kernel_library(
    name = "topk_op",
    srcs = [
      "BatchTopKOnRT_kernel.cc",
      "BlazeTopK_kernel.cc",
      "GTopKV2_kernel.cc",
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
