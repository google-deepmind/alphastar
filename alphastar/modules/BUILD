load("@my_deps//:requirements.bzl", "requirement")
load("//alphastar:pytype.bzl", "pytype_strict_contrib_test", "pytype_strict_library")
load("//alphastar:strict.bzl", "py_strict_test")

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

pytype_strict_library(
    name = "types",
    srcs = ["types.py"],
    srcs_version = "PY3",
    deps = [requirement("dm-acme")],
)

pytype_strict_library(
    name = "optimizers",
    srcs = ["optimizers.py"],
    srcs_version = "PY3",
    deps = [
        requirement("dm-haiku"),
        requirement("jax"),
        requirement("jaxlib"),
        requirement("numpy"),
        requirement("optax"),
        "@absl_py//absl/logging",
    ],
)

py_strict_test(
    name = "optimizers_test",
    srcs = ["optimizers_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":optimizers",
        requirement("chex"),
        requirement("dm-haiku"),
        requirement("jax"),
        requirement("jaxlib"),
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)

pytype_strict_library(
    name = "common",
    srcs = ["common.py"],
    srcs_version = "PY3",
    deps = [
        requirement("dm-acme"),
        "//alphastar/collections:jax",
        requirement("chex"),
        requirement("ml_collections"),
        requirement("pandas"),
        requirement("tensorflow"),
        "@absl_py//absl/logging",
    ],
)

py_strict_test(
    name = "common_test",
    timeout = "long",
    srcs = ["common_test.py"],
    deps = [
        ":common",
        ":optimizers",
        requirement("dm-acme"),
        "//alphastar/architectures",
        "//alphastar/commons:jax_utils",
        "//alphastar/unplugged/configs",
        "//alphastar/unplugged/data:data_source",
        "//alphastar/unplugged/losses:__init__",
        "//alphastar/unplugged/modules:learner",
        requirement("chex"),
        requirement("jax"),
        requirement("jaxlib"),
        "@absl_py//absl/testing:absltest",
    ],
)

pytype_strict_library(
    name = "evaluator_base",
    srcs = ["evaluator_base.py"],
    srcs_version = "PY3",
    deps = [
        "//alphastar/commons:log_utils",
        requirement("chex"),
        "@dm_env_archive//:dm_env",
    ],
)

pytype_strict_library(
    name = "run_loop",
    srcs = ["run_loop.py"],
    deps = [
        ":evaluator_base",
        "//alphastar/loggers:episode_logger",
        "//alphastar/types:__init__",
        "@dm_env_archive//:dm_env",
        requirement("jax"),
        requirement("jaxlib"),
        requirement("numpy"),
    ],
)

py_strict_test(
    name = "match_generator_test",
    srcs = ["match_generator_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":match_generator",
        "@absl_py//absl/testing:absltest",
        "@pysc2_archive//pysc2/env:sc2_env",
    ],
)

pytype_strict_contrib_test(
    name = "evaluator_test",
    timeout = "eternal",  # TODO(b/200647364)
    srcs = ["evaluator_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "guitar",
        "manual",
        "notap",
        "nozapfhahn",  # ignore from coverage tests
        "optonly",
    ],
    deps = [
        ":evaluator",
        "//alphastar/architectures",
        "//alphastar/loggers:eval_episode_logger",
        "//alphastar/unplugged/configs",  # build_cleaner: keep
        requirement("jax"),
        requirement("jaxlib"),
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "agent_test",
    srcs = ["agent_test.py"],
    shard_count = 4,
    deps = [
        ":agent",
        "//alphastar/architectures",
        "//alphastar/commons:jax_utils",
        "//alphastar/unplugged/data:util",
        "@dm_env_archive//:specs",
        requirement("jax"),
        requirement("jaxlib"),
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)

pytype_strict_contrib_test(
    name = "evaluator_test_internal",
    timeout = "eternal",  # TODO(b/200647364)
    srcs = ["evaluator_test_internal.py"],
    python_version = "PY3",
    shard_count = 3,
    srcs_version = "PY3",
    tags = [
        "guitar",
        "manual",
        "notap",
        "nozapfhahn",  # ignore from coverage tests
        "optonly",
    ],
    deps = [
        ":evaluator",
        "//alphastar/architectures",
        "//alphastar/loggers:eval_episode_logger",
        "//alphastar/unplugged/configs",  # build_cleaner: keep
        requirement("jax"),
        requirement("jaxlib"),
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)

pytype_strict_library(
    name = "match_generator",
    srcs = ["match_generator.py"],
    deps = [
        "//alphastar/commons:competitors",
        "@absl_py//absl/logging",
        "@pysc2_archive//pysc2/env:sc2_env",
    ],
)

pytype_strict_library(
    name = "evaluator",
    srcs = ["evaluator.py"],
    srcs_version = "PY3",
    deps = [
        ":agent",
        ":common",
        ":evaluator_base",
        ":match_generator",
        ":run_loop",
        requirement("dm-acme"),
        "//alphastar/architectures:modular",
        "//alphastar/commons:competitors",
        "//alphastar/commons:jax_utils",
        "//alphastar/commons:log_utils",
        "//alphastar/loggers:episode_logger",
        "//alphastar/types:__init__",
        "//alphastar/unplugged/data:util",
        "//alphastar/unplugged/modules:learner",  # build_cleaner: keep
        requirement("chex"),
        "@dm_env_archive//:dm_env",
        requirement("dm-haiku"),
        requirement("jax"),
        requirement("jaxlib"),
        requirement("numpy"),
        "@absl_py//absl/logging",
        "@pysc2_archive//pysc2/env:converted_env",
        "@pysc2_archive//pysc2/env:sc2_env",
        "@pysc2_archive//pysc2/env/converter:derive_interface_options",
        "@pysc2_archive//pysc2/env/converter/proto:converter_py_pb2",
        "@pysc2_archive//pysc2/maps",
    ],
)

pytype_strict_library(
    name = "agent",
    srcs = ["agent.py"],
    deps = [
        "//alphastar/architectures:modular",
        "//alphastar/types:__init__",
        "//alphastar/unplugged/data:util",
        requirement("dm-haiku"),
        requirement("jax"),
        requirement("jaxlib"),
        "@absl_py//absl/logging",
    ],
)
