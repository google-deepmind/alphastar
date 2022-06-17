load("@my_deps//:requirements.bzl", "requirement")
load("//alphastar:pytype.bzl", "pytype_strict_library")
load("//alphastar:strict.bzl", "py_strict_test")

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

py_strict_test(
    name = "learner_test",
    srcs = ["learner_test.py"],
    shard_count = 2,
    deps = [
        ":learner",
        "//alphastar/architectures",
        "//alphastar/commons:jax_utils",
        "//alphastar/modules:optimizers",
        "//alphastar/unplugged/configs",
        "//alphastar/unplugged/data:data_source",
        "//alphastar/unplugged/data:data_source_base",
        "//alphastar/unplugged/losses:__init__",
        requirement("jax"),
        requirement("jaxlib"),
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)

pytype_strict_library(
    name = "learner",
    srcs = ["learner.py"],
    srcs_version = "PY3",
    deps = [
        requirement("dm-acme"),
        "//alphastar/architectures:modular",
        "//alphastar/commons:log_utils",
        "//alphastar/commons:metrics",
        "//alphastar/modules:agent",
        "//alphastar/modules:common",
        "//alphastar/unplugged/data:data_source_base",
        "//alphastar/unplugged/data:util",
        "//alphastar/unplugged/losses:__init__",
        requirement("chex"),
        requirement("dm-haiku"),
        requirement("jax"),
        requirement("jaxlib"),
        requirement("numpy"),
        requirement("optax"),
        requirement("dm-reverb"),
        requirement("dm-tree"),
        "@absl_py//absl/logging",
    ],
)

pytype_strict_library(
    name = "__init__",
    srcs = ["__init__.py"],
)
