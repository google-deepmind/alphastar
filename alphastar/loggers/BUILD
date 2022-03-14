"""Loggers used in Alphastar training and evaluation."""

load("@my_deps//:requirements.bzl", "requirement")
load("//alphastar:pytype.bzl", "pytype_strict_library")

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

pytype_strict_library(
    name = "episode_logger",
    srcs = ["episode_logger.py"],
    srcs_version = "PY3",
    deps = [
        "//alphastar/commons:log_utils",
        "//alphastar/types:__init__",
        "@dm_env_archive//:dm_env",
    ],
)

pytype_strict_library(
    name = "eval_episode_logger",
    srcs = ["eval_episode_logger.py"],
    deps = [
        ":episode_logger",
        requirement("dm-acme"),
        "//alphastar/commons:log_utils",
        "//alphastar/commons:metrics",
        "//alphastar/modules:common",
        "//alphastar/types:__init__",
        "@dm_env_archive//:dm_env",
        requirement("jax"),
        requirement("jaxlib"),
        requirement("numpy"),
    ],
)
