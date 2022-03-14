load("@my_deps//:requirements.bzl", "requirement")
load("//alphastar:pytype.bzl", "pytype_strict_library")

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

pytype_strict_library(
    name = "architectures",
    srcs = ["architectures.py"],
    deps = [
        ":modular",
        "//alphastar/architectures/dummy",
        "//alphastar/architectures/standard",
        "//alphastar/architectures/standard:configs",
        requirement("frozendict"),
    ],
)

pytype_strict_library(
    name = "modular",
    srcs = ["modular.py"],
    deps = [
        "//alphastar/commons:log_utils",
        "//alphastar/types:__init__",
        requirement("chex"),
        requirement("dm-haiku"),
        requirement("jax"),
        requirement("jaxlib"),
    ],
)

pytype_strict_library(
    name = "util",
    srcs = ["util.py"],
    visibility = [
        "//visibility:public",
        "//alphastar/architectures/dummy:__pkg__",
        "//alphastar/architectures/standard:__pkg__",
    ],
    deps = ["//alphastar/types:__init__"],
)
