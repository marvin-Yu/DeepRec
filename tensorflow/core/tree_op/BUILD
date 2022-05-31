load(
    "//tensorflow:tensorflow.bzl",
    "tf_kernel_library",
    "tf_copts",
)

tf_kernel_library(
    name = "tree_op",
    srcs = [
      "BatchGatherOnRT_kernel.cc",
      "BatchConcatOnRT_kernel.cc",
      "SplitsGather_kernel.cc",
      "GroupGather_kernel.cc",
      "tree_kernel.cc",
      #"FarmHash.cpp",
      #"FarmHash.h",
      "set_kernel.cc"
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core/kernels:eigen_helpers",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/strings",
        "@farmhash_archive//:farmhash",
     ],
)
