load("@my_deps//:requirements.bzl", "requirement")
load("//alphastar:pytype.bzl", "pytype_strict_library")
load("//alphastar:strict.bzl", "py_strict_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_strict_test(
    name = "common_test",
    srcs = ["common_test.py"],
    shard_count = 5,
    deps = [
        ":common",
        ":test_utils",
        "//alphastar/architectures:util",
        "//alphastar/commons:sample",
        "//alphastar/types:__init__",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "merge_test",
    srcs = ["merge_test.py"],
    shard_count = 5,
    deps = [
        ":merge",
        ":test_utils",
        ":units",
        "//alphastar/architectures:modular",
        "//alphastar/types:__init__",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
        requirement("jax"),
        requirement("jaxlib"),
    ],
)

py_strict_test(
    name = "units_test",
    size = "large",
    srcs = ["units_test.py"],
    shard_count = 5,
    deps = [
        ":common",
        ":merge",
        ":test_utils",
        ":units",
        ":util",
        "//alphastar/architectures:modular",
        "//alphastar/architectures/components/static_data:unit_encoder_data",
        "//alphastar/commons:sample",
        "//alphastar/types:__init__",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
        "@dm_env_archive//:specs",
        requirement("jax"),
        requirement("jaxlib"),
        requirement("numpy"),
    ],
)

py_strict_test(
    name = "vector_test",
    srcs = ["vector_test.py"],
    shard_count = 5,
    deps = [
        ":test_utils",
        ":vector",
        "//alphastar/types:__init__",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
        requirement("chex"),
        requirement("jax"),
        requirement("jaxlib"),
    ],
)

py_strict_test(
    name = "visual_test",
    srcs = ["visual_test.py"],
    shard_count = 5,
    deps = [
        ":test_utils",
        ":visual",
        "//alphastar/architectures:util",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
        requirement("chex"),
        requirement("jax"),
        requirement("jaxlib"),
    ],
)

pytype_strict_library(
    name = "common",
    srcs = ["common.py"],
    srcs_version = "PY3",
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":util",
        "//alphastar/architectures:modular",
        "//alphastar/commons:sample",
        "//alphastar/types:__init__",
        "@dm_env_archive//:dm_env",
        "@dm_env_archive//:specs",
        requirement("jax"),
        requirement("jaxlib"),
    ],
)

pytype_strict_library(
    name = "merge",
    srcs = ["merge.py"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":util",
        "//alphastar/architectures:modular",
        "//alphastar/types:__init__",
        "@dm_env_archive//:specs",
        requirement("dm-haiku"),
        requirement("jax"),
        requirement("jaxlib"),
    ],
)

pytype_strict_library(
    name = "test_utils",
    srcs = ["test_utils.py"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//alphastar/architectures:modular",
        "//alphastar/types:__init__",
        "//alphastar/unplugged/data:util",
        "@absl_py//absl/testing:parameterized",
        "@dm_env_archive//:specs",
        requirement("dm-haiku"),
        requirement("jax"),
        requirement("jaxlib"),
    ],
)

pytype_strict_library(
    name = "units",
    srcs = [
        "units.py",
    ],
    data = ["@pysc2_archive//pysc2/env/converter/cc/game_data/python:uint8_lookup.so"],
    srcs_version = "PY3",
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":util",
        "//alphastar/architectures:modular",
        "//alphastar/architectures/components/static_data:camera_masks",
        "//alphastar/architectures/components/static_data:unit_encoder_data",
        "//alphastar/commons:sample",
        "//alphastar/types:__init__",
        requirement("chex"),
        "@dm_env_archive//:specs",
        requirement("dm-haiku"),
        requirement("jax"),
        requirement("jaxlib"),
        requirement("numpy"),
        "@s2client_proto//s2clientprotocol:raw_py_pb2",
        "@pysc2_archive//pysc2/env/converter/cc/game_data/python:__init__",
        "@pysc2_archive//pysc2/lib:features",
    ],
)

pytype_strict_library(
    name = "util",
    srcs = ["util.py"],
    srcs_version = "PY3",
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//alphastar/types:__init__",
        requirement("chex"),
        requirement("dm-haiku"),
        requirement("jax"),
        requirement("jaxlib"),
        requirement("numpy"),
        "@pysc2_archive//pysc2/lib:actions",
    ],
)

# The tests in this package that are tagged dmfast_stable are excluded from
# sanitizers, since they would need to have a longer timeout to succeed under
# sanitizers. But a longer timeout would disqualify the tests from being part of
# deepmind.research.fast.stable, which is more important that running
# sanitizers.

pytype_strict_library(
    name = "vector",
    srcs = ["vector.py"],
    srcs_version = "PY3",
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":util",
        "//alphastar/architectures:modular",
        "//alphastar/types:__init__",
        requirement("chex"),
        "@dm_env_archive//:specs",
        requirement("dm-haiku"),
        requirement("jax"),
        requirement("jaxlib"),
        requirement("numpy"),
    ],
)

pytype_strict_library(
    name = "visual",
    srcs = ["visual.py"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":util",
        "//alphastar/architectures:modular",
        "//alphastar/architectures/components/static_data:camera_masks",
        "//alphastar/commons:sample",
        "//alphastar/types:__init__",
        requirement("chex"),
        "@dm_env_archive//:specs",
        requirement("dm-pix"),
        requirement("dm-haiku"),
        requirement("jax"),
        requirement("jaxlib"),
        requirement("numpy"),
    ],
)

pytype_strict_library(
    name = "__init__",
    srcs = ["__init__.py"],
)
