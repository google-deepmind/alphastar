# Package for custom data structures.

load("@my_deps//:requirements.bzl", "requirement")
load("//alphastar:strict.bzl", "py_library_with_strictdeps_info", "py_strict_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_library_with_strictdeps_info(
    name = "collections",
    srcs = [
        "__init__.py",
        "structure.py",
    ],
    srcs_version = "PY3",
)

py_strict_test(
    name = "structure_test",
    srcs = ["structure_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":collections",
        "@absl_py//absl/testing:absltest",
        requirement("dm-tree"),
    ],
)

py_library(
    name = "jax",
    srcs = ["jax.py"],
    srcs_version = "PY3",
    deps = [
        ":collections",
        "@absl_py//absl/logging",
        requirement("jax"),
        requirement("jaxlib"),
    ],
)
