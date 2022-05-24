load("@my_deps//:requirements.bzl", "requirement")
load("//alphastar:strict.bzl", "py_strict_test")
load("//alphastar:pytype.bzl", "pytype_strict_library")

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

pytype_strict_library(
    name = "data_source_base",
    srcs = ["data_source_base.py"],
    deps = [
        "//alphastar/collections",
        "//alphastar/types:__init__",
        "@pysc2_archive//pysc2/env:enums",
    ],
)

pytype_strict_library(
    name = "util",
    srcs = ["util.py"],
    srcs_version = "PY3",
    deps = [
        "//alphastar/types:__init__",
        "@absl_py//absl/logging",
        requirement("apache-beam"),
        requirement("chex"),
        "@dm_env_archive//:dm_env",
        "@dm_env_archive//:specs",
        requirement("jax"),
        requirement("jaxlib"),
        requirement("numpy"),
        requirement("tensorflow"),
        requirement("tensorflow-datasets"),
        requirement("dm-tree"),
    ],
)

pytype_strict_library(
    name = "data_source",
    srcs = ["data_source.py"],
    srcs_version = "PY3",
    deps = [
        ":data_source_base",
        ":util",
        "//alphastar/types:__init__",
        "//alphastar/unplugged/data:path_utils",
        "@pysc2_archive//pysc2/env:converted_env",
        "@pysc2_archive//pysc2/env:enums",
        "@pysc2_archive//pysc2/env/converter/proto:converter_py_pb2",
        requirement("tensorflow"),
    ],
)

py_binary(
    name = "generate_dataset",
    srcs = ["generate_dataset.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":util",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
        "@absl_py//absl/logging",
        "@com_google_protobuf//:protobuf_python",
        requirement("chex"),
        "@dm_env_archive//:dm_env",
        requirement("jax"),
        requirement("jaxlib"),
        requirement("numpy"),
        "@pysc2_archive//pysc2/env:converted_env",
        "@pysc2_archive//pysc2/env/converter/proto:converter_py_pb2",
        "@pysc2_archive//pysc2/lib:protocol",
        "@pysc2_archive//pysc2/lib:remote_controller",
        "@pysc2_archive//pysc2/lib:sc_process",
        "@pysc2_archive//pysc2/lib/replay:replay_converter",
        requirement("tensorflow"),
        requirement("websocket-client"),
    ],
)

py_binary(
    name = "generate_partitions",
    srcs = ["generate_partitions.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
        "@absl_py//absl/logging",
        requirement("tensorflow"),
    ],
)

pytype_strict_library(
    name = "path_utils",
    srcs = ["path_utils.py"],
)

pytype_strict_library(
    name = "__init__",
    srcs = ["__init__.py"],
)

py_strict_test(
    name = "util_test",
    srcs = ["util_test.py"],
    python_version = "PY3",
    shard_count = 2,
    srcs_version = "PY3",
    deps = [
        ":util",
        "@absl_py//absl/testing:parameterized",
        requirement("chex"),
        requirement("numpy"),
        requirement("tensorflow"),
        requirement("dm-tree"),
    ],
)
