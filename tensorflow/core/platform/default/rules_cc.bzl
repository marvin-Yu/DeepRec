"""This forwards all of rules_cc's relevant rules under a common file"""

load(
    "@rules_cc//cc:defs.bzl",
    _cc_binary = "cc_binary",
    _cc_import = "cc_import",
    _cc_library = "cc_library",
    _cc_test = "cc_test",
)

cc_binary = _cc_binary
cc_import = _cc_import
cc_library = _cc_library
cc_test = _cc_test