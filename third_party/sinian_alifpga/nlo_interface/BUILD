# Description: This file is for nlo interface

licenses(["notice"])
cc_library(
    name = "nlo_util_header",
    hdrs = [
        "nlo_map_util.h",
        "nlo_iterator_range.h",
        "nlo_iterator_util.h",
        "nlo_status.h",
        "nlo_primitive_type.h",
        "nlo_primitive_util.h",
        "nlo_shape.h",
        "nlo_shape_util.h",
        "nlo_literal_util.h",
        "nlo_name_uniquer.h",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "nlo_header",
    hdrs = [
        "dfs_nlo_visitor.h",
        "dfs_nlo_visitor_with_default.h",
        "nlo_computation.h",
        "nlo_instruction.h",
        "nlo_module.h",
        "nlo_opcode.h",
        "nlo_reachability.h",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "sinian_alifpga_nlo_opt_header",
    hdrs = [
     "sinian_alifpga_nlo_opt.h",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "nlo_interface",
    deps = [
        ":nlo_util_header",
        ":nlo_header",
        ":sinian_alifpga_nlo_opt_header",
    ],
    visibility = ["//visibility:public"],
)
