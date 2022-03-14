load("@my_deps//:requirements.bzl", "requirement")
load("//alphastar:pytype.bzl", "pytype_strict_library")

licenses(["notice"])

package(
    default_visibility = [
        "//alphastar/architectures:__subpackages__",
    ],
)

pytype_strict_library(
    name = "camera_masks",
    srcs = ["camera_masks.py"],
    deps = [
        requirement("numpy"),
        "@pysc2_archive//pysc2/lib:actions",
    ],
)

pytype_strict_library(
    name = "unit_encoder_data",
    srcs = ["unit_encoder_data.py"],
    data = ["@pysc2_archive//pysc2/env/converter/cc/game_data/python:uint8_lookup.so"],
    deps = [
        requirement("chex"),
        requirement("numpy"),
        "@s2client_proto//s2clientprotocol:raw_py_pb2",
        "@pysc2_archive//pysc2/lib:actions",
        "@pysc2_archive//pysc2/lib:features",
        "@pysc2_archive//pysc2/lib:units",
    ],
)
