load("@my_deps//:requirements.bzl", "requirement")
load("//alphastar:pytype.bzl", "pytype_strict_library")
load("//alphastar:strict.bzl", "py_strict_test")

licenses(["notice"])

package(
    default_visibility = [
        "//visibility:public",
    ],
)

pytype_strict_library(
    name = "sample",
    srcs = ["sample.py"],
    deps = [
        requirement("chex"),
        requirement("dm-haiku"),
        requirement("jax"),
        requirement("jaxlib"),
    ],
)

pytype_strict_library(
    name = "log_utils",
    srcs = ["log_utils.py"],
    deps = [
        "//alphastar/collections",
        requirement("chex"),
        requirement("jax"),
        requirement("jaxlib"),
    ],
)

pytype_strict_library(
    name = "metrics",
    srcs = ["metrics.py"],
    srcs_version = "PY3",
    deps = [
        ":log_utils",
        requirement("chex"),
        requirement("jax"),
        requirement("jaxlib"),
    ],
)

pytype_strict_library(
    name = "jax_utils",
    srcs = ["jax_utils.py"],
    srcs_version = "PY3",
    deps = [
        requirement("jax"),
        requirement("jaxlib"),
    ],
)

py_strict_test(
    name = "jax_utils_test",
    srcs = ["jax_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":jax_utils",
        "@absl_py//absl/testing:absltest",
        requirement("jax"),
        requirement("jaxlib"),
    ],
)

pytype_strict_library(
    name = "competitors",
    srcs = ["competitors.py"],
    deps = [
        requirement("numpy"),
        "@s2client_proto//s2clientprotocol:common_py_pb2",
        "@s2client_proto//s2clientprotocol:sc2api_py_pb2",
    ],
)
