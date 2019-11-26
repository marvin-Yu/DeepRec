load(
    "//tensorflow:tensorflow.bzl",
    "tf_kernel_library",
    "tf_copts",
)

cc_library(
    name = "take_ops_op_lib",
    srcs = [
        "take_op.cc",
        "take_axis_op.cc",
    ],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:framework",
        ":take_ops",
    ],
    alwayslink = 1,
)

tf_kernel_library(
    name = "take_ops",
    srcs = [
        "take_lib.h",
        "take_lib_cpu.h",
        "take_lib_cpu.cc",
        "take.cc",
        "take_grad.cc",
        "take_grad_lib.h",
        "take_grad_lib_cpu.h",
        "take_grad_lib_cpu.cc",
        "take_axis_lib.h",
        "take_axis_lib_cpu.h",
        "take_axis_lib_cpu.cc",
        "take_axis.cc",
    ],
    gpu_srcs = [
        "take_lib.h",
        "take_lib_gpu.cu.cc",
        "take_lib_gpu_impl.cu.cc",
        "take_grad_lib_gpu.cu.cc",
        "take_grad_lib_gpu_impl.cu.cc",
        "take_axis_lib.h",
        "take_axis_lib_gpu.cu.cc",
        "take_axis_lib_gpu_impl.cu.cc",
    ],
    deps = [
        "//tensorflow/core/kernels:gpu_device_array",
        "//tensorflow/core/kernels:eigen_helpers",
    ],
)
