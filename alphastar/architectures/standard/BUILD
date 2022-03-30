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
    name = "configs",
    srcs = glob(["configs/*.py"]),
    deps = [
        "//alphastar/architectures/components:merge",
        "//alphastar/commons:sample",
        requirement("jax"),
        requirement("jaxlib"),
        requirement("ml_collections"),
    ],
)

pytype_strict_library(
    name = "encoders",
    srcs = ["encoders.py"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//alphastar/architectures:modular",
        "//alphastar/architectures:util",
        "//alphastar/architectures/components:common",
        "//alphastar/architectures/components:merge",
        "//alphastar/architectures/components:units",
        "//alphastar/architectures/components:vector",
        "//alphastar/architectures/components:visual",
        "//alphastar/types:__init__",
        requirement("jax"),
        requirement("jaxlib"),
        requirement("ml_collections"),
    ],
)

pytype_strict_library(
    name = "heads",
    srcs = ["heads.py"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//alphastar/architectures:modular",
        "//alphastar/architectures:util",
        "//alphastar/architectures/components:common",
        "//alphastar/architectures/components:merge",
        "//alphastar/architectures/components:units",
        "//alphastar/architectures/components:vector",
        "//alphastar/architectures/components:visual",
        "//alphastar/types:__init__",
        requirement("jax"),
        requirement("jaxlib"),
        requirement("ml_collections"),
    ],
)

py_strict_test(
    name = "standard_test",
    timeout = "long",
    srcs = ["standard_test.py"],
    shard_count = 2,
    deps = [
        ":standard",
        "//alphastar/architectures/components:test_utils",
        "//alphastar/architectures/standard:configs",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)

pytype_strict_library(
    name = "standard",
    srcs = ["standard.py"],
    deps = [
        ":encoders",
        ":heads",
        ":torso",
        "//alphastar/architectures:modular",
        "//alphastar/architectures:util",
        "//alphastar/types:__init__",
        requirement("ml_collections"),
    ],
)

pytype_strict_library(
    name = "torso",
    srcs = ["torso.py"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//alphastar/architectures:modular",
        "//alphastar/architectures:util",
        "//alphastar/architectures/components:merge",
        "//alphastar/architectures/components:units",
        "//alphastar/architectures/components:vector",
        "//alphastar/architectures/components:visual",
        "//alphastar/types:__init__",
        requirement("ml_collections"),
    ],
)
