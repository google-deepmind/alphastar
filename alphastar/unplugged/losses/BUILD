load("@my_deps//:requirements.bzl", "requirement")
load("//alphastar:strict.bzl", "py_strict_test")
load("//alphastar:pytype.bzl", "pytype_strict_library")

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

pytype_strict_library(
    name = "util",
    srcs = ["util.py"],
    deps = [
        "//alphastar/commons:log_utils",
        "//alphastar/types:__init__",
        requirement("chex"),
        "@dm_env_archive//:dm_env",
        requirement("jax"),
        requirement("jaxlib"),
        requirement("numpy"),
    ],
)

pytype_strict_library(
    name = "supervised",
    srcs = ["supervised.py"],
    deps = [
        ":loss_base",
        ":util",
        "//alphastar/architectures:util",
        "//alphastar/commons:log_utils",
        "//alphastar/types:__init__",
        requirement("chex"),
        "@dm_env_archive//:specs",
        requirement("jax"),
        requirement("jaxlib"),
    ],
)

pytype_strict_library(
    name = "__init__",
    srcs = ["__init__.py"],
    deps = [
        ":loss_base",
        ":supervised",
    ],
)

pytype_strict_library(
    name = "loss_base",
    srcs = ["loss_base.py"],
    deps = [
        "//alphastar/commons:log_utils",
        "//alphastar/types:__init__",
        requirement("chex"),
        requirement("jax"),
        requirement("jaxlib"),
    ],
)

py_strict_test(
    name = "supervised_test",
    srcs = ["supervised_test.py"],
    deps = [
        ":supervised",
        "//alphastar/commons:log_utils",
        "//alphastar/types:__init__",
        "@absl_py//absl/testing:absltest",
        "@dm_env_archive//:dm_env",
        "@dm_env_archive//:specs",
        requirement("jax"),
        requirement("jaxlib"),
        requirement("numpy"),
    ],
)
