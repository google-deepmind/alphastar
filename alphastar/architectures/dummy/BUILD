load("@my_deps//:requirements.bzl", "requirement")
load("//alphastar:pytype.bzl", "pytype_strict_library")
load("//alphastar:strict.bzl", "py_strict_test")

licenses(["notice"])

package(
    default_visibility = [
        "//visibility:public",
    ],
)

py_strict_test(
    name = "dummy_test",
    srcs = ["dummy_test.py"],
    deps = [
        ":dummy",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
        "@dm_env_archive//:specs",
        requirement("dm-haiku"),
        requirement("jax"),
        requirement("jaxlib"),
    ],
)

pytype_strict_library(
    name = "dummy",
    srcs = ["dummy.py"],
    deps = [
        "//alphastar/architectures:modular",
        "//alphastar/architectures:util",
        "//alphastar/architectures/components:common",
        "//alphastar/architectures/components:units",
        "//alphastar/architectures/components:vector",
        "//alphastar/commons:sample",
        "//alphastar/types:__init__",
        requirement("jax"),
        requirement("jaxlib"),
    ],
)
