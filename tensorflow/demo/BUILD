load("//tensorflow:tensorflow.bzl", "tf_cc_binary")
tf_cc_binary(
     name = "demo", #目标文件名
     srcs = ["call.cc"], #源代码文件名
     deps = [
         "//tensorflow/cc:cc_ops",
         "//tensorflow/cc:client_session",
         "//tensorflow/core:tensorflow",
         "//tensorflow/c:c_api_experimental",
         "//tensorflow/c:c_api_blaze",
         "//tensorflow/compiler/jit:flags"
         ],
    linkopts = ["-L/usr/local/cuda/lib64 -lnvToolsExt"],
 )
