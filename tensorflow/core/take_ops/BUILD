load(
    "//tensorflow:tensorflow.bzl",
    "tf_kernel_library",
    "tf_copts",
)

load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")

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
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core/kernels:eigen_helpers",
    ] + if_cuda([
        "//tensorflow/core/kernels:gpu_device_array",
    ]),
)
