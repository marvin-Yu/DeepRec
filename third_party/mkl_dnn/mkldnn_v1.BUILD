exports_files(["LICENSE"])

load(
    "@org_tensorflow//third_party/mkl:build_defs.bzl",
    "if_mkl",
)

load(
    "@org_tensorflow//tensorflow:tensorflow.bzl",
    "tf_openmp_copts",
)

load(
    "@org_tensorflow//third_party/mkl_dnn:build_defs.bzl",
    "if_mkl_open_source_only",
    "if_mkldnn_threadpool",
)

load(
    "@org_tensorflow//third_party/mkl:build_defs.bzl",
    "if_mkl_ml",
)

load(
    "@org_tensorflow//third_party:common.bzl",
    "template_rule",
)

config_setting(
    name = "clang_linux_x86_64",
    values = {
        "cpu": "k8",
        "define": "using_clang=true",
    },
)

_DNNL_RUNTIME_OMP = {
    "#cmakedefine DNNL_CPU_THREADING_RUNTIME DNNL_RUNTIME_${DNNL_CPU_THREADING_RUNTIME}": "#define DNNL_CPU_THREADING_RUNTIME DNNL_RUNTIME_OMP",
    "#cmakedefine DNNL_CPU_RUNTIME DNNL_RUNTIME_${DNNL_CPU_RUNTIME}": "#define DNNL_CPU_RUNTIME DNNL_RUNTIME_OMP",
    "#cmakedefine DNNL_GPU_RUNTIME DNNL_RUNTIME_${DNNL_GPU_RUNTIME}": "#define DNNL_GPU_RUNTIME DNNL_RUNTIME_NONE",
}

_DNNL_RUNTIME_THREADPOOL = {
    "#cmakedefine DNNL_CPU_THREADING_RUNTIME DNNL_RUNTIME_${DNNL_CPU_THREADING_RUNTIME}": "#define DNNL_CPU_THREADING_RUNTIME DNNL_RUNTIME_THREADPOOL",
    "#cmakedefine DNNL_CPU_RUNTIME DNNL_RUNTIME_${DNNL_CPU_RUNTIME}": "#define DNNL_CPU_RUNTIME DNNL_RUNTIME_THREADPOOL",
    "#cmakedefine DNNL_GPU_RUNTIME DNNL_RUNTIME_${DNNL_GPU_RUNTIME}": "#define DNNL_GPU_RUNTIME DNNL_RUNTIME_NONE",
}

_DNNL_RUNTIME_SEQ = {
    "#cmakedefine DNNL_CPU_THREADING_RUNTIME DNNL_RUNTIME_${DNNL_CPU_THREADING_RUNTIME}": "#define DNNL_CPU_THREADING_RUNTIME DNNL_RUNTIME_SEQ",
    "#cmakedefine DNNL_CPU_RUNTIME DNNL_RUNTIME_${DNNL_CPU_RUNTIME}": "#define DNNL_CPU_RUNTIME DNNL_RUNTIME_SEQ",
    "#cmakedefine DNNL_GPU_RUNTIME DNNL_RUNTIME_${DNNL_GPU_RUNTIME}": "#define DNNL_GPU_RUNTIME DNNL_RUNTIME_NONE",
}

template_rule(
    name = "dnnl_config_h",
    src = "include/dnnl_config.h.in",
    out = "include/dnnl_config.h",
    substitutions = select({
        "@org_tensorflow//third_party/mkl_dnn:build_with_mkldnn_threadpool": _DNNL_RUNTIME_THREADPOOL,
        "@org_tensorflow//third_party/mkl:build_with_mkl": _DNNL_RUNTIME_OMP,
        "//conditions:default": _DNNL_RUNTIME_SEQ,
    }),
)

# Create the file dnnl_version.h with DNNL version numbers.
# Currently, the version numbers are hard coded here. If DNNL is upgraded then
# the version numbers have to be updated manually. The version numbers can be
# obtained from the PROJECT_VERSION settings in CMakeLists.txt. The variable is
# set to "version_major.version_minor.version_patch". The git hash version can
# be set to NA.
# TODO(agramesh1): Automatically get the version numbers from CMakeLists.txt.
template_rule(
    name = "dnnl_version_h",
    src = "include/dnnl_version.h.in",
    out = "include/dnnl_version.h",
    substitutions = {
        "@DNNL_VERSION_MAJOR@": "1",
        "@DNNL_VERSION_MINOR@": "8",
        "@DNNL_VERSION_PATCH@": "1",
        "@DNNL_VERSION_HASH@": "N/A",
    },
)

cc_library(
    name = "mkl_dnn",
    srcs = glob([
        "src/common/*.cpp",
        "src/common/*.hpp",
        "src/cpu/*.cpp",
        "src/cpu/*.hpp",
        "src/cpu/gemm/**/*.cpp",
        "src/cpu/gemm/**/*.hpp",
        "src/cpu/matmul/**/*.cpp",
        "src/cpu/matmul/**/*.hpp",
        "src/cpu/rnn/**/*.cpp",
        "src/cpu/rnn/**/*.hpp",
        "src/cpu/x64/**/*.cpp",
        "src/cpu/x64/**/*.hpp",
        "src/cpu/xbyak/*.h",
        "src/cpu/x64/jit_utils/jitprofiling/*.c",
        "src/cpu/x64/jit_utils/jitprofiling/*.h",
    ]) + [
        ":dnnl_config_h",
        ":dnnl_version_h",
    ],
    hdrs = glob(["include/*"]),
    copts = [
        "-fexceptions",
        "-UUSE_MKL",
        "-UUSE_CBLAS",
    ] + select({
        "@org_tensorflow//tensorflow:linux_x86_64": [
            "-fopenmp",  # only works with gcc
        ],
        # TODO(ibiryukov): enable openmp with clang by including libomp as a
        # dependency.
        ":clang_linux_x86_64": [],
        "//conditions:default": [],
    }),
    includes = [
        "include",
        "src",
        "src/common",
        "src/cpu",
        "src/cpu/gemm",
        "src/cpu/xbyak",
    ],
    visibility = ["//visibility:public"],
    deps = if_mkl_ml(
        ["@org_tensorflow//third_party/mkl:intel_binary_blob"],
        [],
    ),
)

cc_library(
    name = "dnnl_single_threaded",
    srcs = glob([
        "src/common/*.cpp",
        "src/cpu/*.cpp",
        "src/cpu/gemm/**/*.cpp",
        "src/cpu/matmul/**/*.cpp",
        "src/cpu/rnn/**/*.cpp",
        "src/cpu/x64/**/*.cpp",
        "src/cpu/x64/jit_utils/jitprofiling/*.c",
    ]) + [
        ":dnnl_config_h",
        ":dnnl_version_h",
    ],
    copts = [
        "-fexceptions",
        "-DDNNL_ENABLE_MAX_CPU_ISA",
    ],
    includes = [
        "include",
        "src",
        "src/common",
        "src/cpu",
        "src/cpu/gemm",
        "src/cpu/gemm/f32",
        "src/cpu/gemm/s8x8s32",
        "src/cpu/matmul",
        "src/cpu/rnn",
        "src/cpu/x64",
        "src/cpu/x64/jit_utils",
        "src/cpu/x64/jit_utils/jitprofiling",
        "src/cpu/x64/xbyak",
    ],
    textual_hdrs = glob([
        "include/*",
        "src/common/*.hpp",
        "src/cpu/*.hpp",
        "src/cpu/**/*.hpp",
        "src/cpu/xbyak/*.h",
    ]),
    visibility = ["//visibility:public"],
)
