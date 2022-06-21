workspace(name = "alphastar")

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "rules_python",
    url = "https://github.com/bazelbuild/rules_python/releases/download/0.4.0/rules_python-0.4.0.tar.gz",
    sha256 = "954aa89b491be4a083304a2cb838019c8b8c3720a7abb9c4cb81ac7a24230cea",
)

load("@rules_python//python:pip.bzl", "pip_install")

# Create a central external repo, @my_deps, that contains Bazel targets for all the
# third-party packages specified in the requirements.txt file.
pip_install(
   name = "my_deps",
   requirements = "@//bazel:requirements.txt",
)

# Needed to define .bzl libs.
http_archive(
    name = "bazel_skylib",
    strip_prefix = "bazel-skylib-main",
    urls = ["https://github.com/bazelbuild/bazel-skylib/archive/main.zip"],
)

# We can't pip install PySC2 (yet).
git_repository(
    name = "pysc2_archive",
    remote = "sso://team/deepmind-sc2/starcraft2",
    branch = "master",
)

# Note: if you wish to use a local repository for PySC2 rather than fetching the
# sources from GitHub (e.g. because you have made local modifications), please
# comment out the section above and uncomment the section below, substituting
# the real path to the local repository. See here for further information:
# https://docs.bazel.build/versions/main/be/workspace.html#local_repository
#
# local_repository(
#     name = "pysc2_archive",
#     path = "/some/path/to/starcraft2",  # Replace with the real repo path
# )

load("@pysc2_archive//bazel:create_external_repos.bzl", "pysc2_create_external_repos")
pysc2_create_external_repos(pysc2_repo_name = "pysc2_archive")

load("@pysc2_archive//bazel:setup_external_repos.bzl", "pysc2_setup_external_repos")
pysc2_setup_external_repos()

bind(
    name = "python_headers",
    actual = "@local_config_python//:python_headers",
)

bind(
    name = "six",
    actual = "@six_archive//:six",
)
