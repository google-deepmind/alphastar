"Configs for Alphastar."

load("@my_deps//:requirements.bzl", "requirement")
load("//alphastar:pytype.bzl", "pytype_strict_library")

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

pytype_strict_library(
    name = "configs",
    srcs = glob(["*.py"]),
    srcs_version = "PY3",
    deps = [
        "//alphastar/architectures",
        "//alphastar/modules:optimizers",
        requirement("ml_collections"),
        "@s2client_proto//s2clientprotocol:common_py_pb2",
        "@pysc2_archive//pysc2/env/converter/proto:converter_py_pb2",
    ],
)
