load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "riegeli",
    sha256 = "419cc9613217c1fe02fc3921a573d4ef9c1e4ee0b7daa5d147d709a2af13a6ae",
    strip_prefix = "riegeli-9ca2ab068edf8396d5364d79705792bfe4ca79d9",
    urls = ["https://github.com/google/riegeli/archive/9ca2ab068edf8396d5364d79705792bfe4ca79d9.tar.gz"],
    repo_mapping = {
        "@net_zstd": "@zstd",
        "@com_google_googleapis": "@googleapis",
        "@com_google_googletest": "@googletest",
        "@com_google_absl": "@abseil-cpp",
    },
)

http_archive(
    name = "tensorstore",
    integrity = "sha256-bn5EENBIZYRdFFzXWEtW7eDFyGAk4FxPuGzAm5Z34F8=",
    strip_prefix = "tensorstore-2268931b0ff74d322b95632d11d8db7d6dda280f",
    urls = ["https://github.com/google/tensorstore/archive/2268931b0ff74d322b95632d11d8db7d6dda280f.tar.gz"],
    repo_mapping = {
        "@grpc": "@com_github_grpc_grpc",
        "@com_google_protobuf": "@com_google_protobuf",
        "@nlohmann_json": "@nlohmann_json",
    },
)

http_archive(
    name = "net_sourceforge_half",
    urls = ["https://sourceforge.net/projects/half/files/half/2.2.1/half-2.2.1.zip"],
    sha256 = "76ddbf406e9d9b772ec73af2bf925b38b290b4390cc4064720a08d4b4bca0aa9",
    build_file_content = """
load("@rules_cc//cc:cc_library.bzl", "cc_library")
cc_library(
    name = "half",
    hdrs = ["include/half.hpp"],
    includes = ["include"],
    visibility = ["//visibility:public"],
)
""",
)

load("@tensorstore//:external.bzl", "tensorstore_dependencies")
tensorstore_dependencies()
