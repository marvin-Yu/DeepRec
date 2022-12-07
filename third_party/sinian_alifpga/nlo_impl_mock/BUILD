# Description:
#   NLO Package.

licenses(["notice"])  # Apache 2.0
cc_library(
    name = "sinian_alifpga_nlo_opt",
    srcs = ["sinian_alifpga_nlo_opt.cc",
            "nlo_module.cc",
            "nlo_instruction.cc",
            "nlo_computation.cc",
            "dfs_nlo_visitor.cc",
            "nlo_name_uniquer.cc",
            "nlo_shape_util.cc",
            "nlo_literal_util.cc",
            "nlo_opcode.cc",],
    deps = [
        "//third_party/sinian_alifpga/nlo_interface:nlo_interface",
        "//third_party/sinian_alifpga:fpga_stream",
    ],
    copts = ["-fPIC"],
    visibility = ["//visibility:public"],
)


