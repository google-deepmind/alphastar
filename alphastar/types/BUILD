"""Custom types used in Alphastar"""

load("@my_deps//:requirements.bzl", "requirement")
load("//alphastar:strict.bzl", "py_strict_test")
load("//alphastar:pytype.bzl", "pytype_strict_library")

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

pytype_strict_library(
    name = "types",
    srcs = ["types.py"],
    deps = [
        requirement("chex"),
        "@dm_env_archive//:specs",
        requirement("jax"),
        requirement("jaxlib"),
        requirement("numpy"),
    ],
)

py_strict_test(
    name = "types_test",
    srcs = ["types_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":__init__",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
        requirement("chex"),
        "@dm_env_archive//:specs",
        requirement("jax"),
        requirement("jaxlib"),
        requirement("numpy"),
        requirement("dm-tree"),
    ],
)

pytype_strict_library(
    name = "__init__",
    srcs = ["__init__.py"],
    deps = [":types"],
)
