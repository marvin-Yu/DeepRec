# Description: This file is for fpga backend

licenses(["notice"])

cc_library(
    name = "sinian_alifpga_lib",
    srcs = ["libsinian_alifpga.so"],
    deps = [
      ":alidla_lib",
    ],
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "libsinian_alifpga.so",
    srcs = [
            "fpga_stream.cc",
            "fpga_runtime.cc",
    ],
    deps = [
        ":fpga_stream",
        ":fpga_runtime",
        "//third_party/sinian_alifpga/nlo_impl_mock:sinian_alifpga_nlo_opt",
    ],
    copts = ["-fPIC"],
    linkshared = 1,
)

cc_library(
    name = "sinian_alifpga_nlo_pass",
    srcs = ["sinian_alifpga_nlo_pass.cc",],
    hdrs = ["sinian_alifpga_nlo_pass.h"],
    deps = [
        ":fpga_stream",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//third_party/sinian_alifpga/nlo_interface:nlo_interface",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "fpga_stream",
    hdrs = ["fpga_stream.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "fpga_runtime",
    hdrs = ["fpga_runtime.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "alidla_lib",
    srcs = ["libAliDLA.so"],
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "libAliDLA.so",
    deps = [
        ":alidla",
    ],
    linkshared = 1,
)

cc_library(
    name = "alidla",
    hdrs = ["alidla.h"],
    srcs = ["alidla.cc"],
    copts = ["-fPIC"],
    alwayslink = 1,
    visibility = ["//visibility:public"],
)

