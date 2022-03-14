load("@bazel_skylib//:bzl_library.bzl", "bzl_library")

licenses(["notice"])

package_group(
    name = "visibility",
    packages = [
        "//alphastar/...",
    ],
)

exports_files(["LICENSE"])

bzl_library(
    name = "pytype",
    srcs = ["pytype.bzl"],
)

bzl_library(
    name = "strict",
    srcs = ["strict.bzl"],
)
