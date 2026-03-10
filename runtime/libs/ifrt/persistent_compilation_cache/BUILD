load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_visibility = [
    "//learning/infra/mira/experimental/orbax_model:__subpackages__",
    "@neptune//:__subpackages__",
])

cc_library(
    name = "ifrt_persistent_compilation_cache",
    srcs = ["ifrt_persistent_compilation_cache.cc"],
    hdrs = ["ifrt_persistent_compilation_cache.h"],
    deps = [
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status:statusor",
        "@xla//xla/pjrt:pjrt_executable",
        "@xla//xla/python/ifrt",
        "@xla//xla/python/ifrt/hlo:hlo_program",
        "@xla//xla/python/pjrt_ifrt:xla_ifrt",
    ],
)
