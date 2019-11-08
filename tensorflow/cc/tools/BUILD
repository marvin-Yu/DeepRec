# Description:
# TensorFlow cc tools.

load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
    "tf_cc_binary",
    "tf_copts",
)

load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["LICENSE"])

cc_library(
    name = "freeze_saved_model",
    srcs = ["freeze_saved_model.cc"],
    hdrs = ["freeze_saved_model.h"],
    deps = [
        "//tensorflow/cc/saved_model:loader",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ],
)

tf_cc_test(
    name = "freeze_saved_model_test",
    srcs = ["freeze_saved_model_test.cc"],
    deps = [
        ":freeze_saved_model",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:resource_variable_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensorflow",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

cc_library(
    name = "graph_editor",
    srcs = ["graph_editor.cc"],
    hdrs = ["graph_editor.h"],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/strings:str_format",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:error_codes_proto_cc",
        "//tensorflow/tools/graph_transforms:transform_graph_main_lib",
    ],
)

tf_cc_binary(
    name = "graph_editor_tool",
    srcs = ["graph_editor_test.cc"],
    copts = tf_copts(),
    linkopts = [] + if_cuda(["-lcuda"]),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        ":graph_editor",
    ],
)