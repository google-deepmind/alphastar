load("@my_deps//:requirements.bzl", "requirement")
load("//alphastar:strict.bzl", "py_strict_binary")

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

py_strict_binary(
    name = "train",
    srcs = ["train.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
        "@absl_py//absl/logging",
        requirement("dm-acme"),
        "//alphastar/architectures",
        "//alphastar/modules:common",
        "//alphastar/modules:optimizers",
        "//alphastar/unplugged/configs",  # build_cleaner: keep
        "//alphastar/unplugged/data:data_source",
        "//alphastar/unplugged/data:data_source_base",
        "//alphastar/unplugged/losses:__init__",
        "//alphastar/unplugged/modules:learner",
        requirement("jax"),
        requirement("jaxlib"),
        requirement("ml_collections"),
    ],
)

py_strict_binary(
    name = "evaluate",
    srcs = ["evaluate.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//alphastar/architectures",
        "//alphastar/loggers:eval_episode_logger",
        "//alphastar/modules:evaluator",
        "//alphastar/unplugged/configs",  # build_cleaner: keep
        "@absl_py//absl:app",
        requirement("ml_collections"),
    ],
)
