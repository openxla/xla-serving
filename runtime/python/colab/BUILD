load("//learning/grp/pynball:build_defs.bzl", "ipynb_deps")
load("//research/colab:build_defs.bzl", "colab_binary", "colab_notebook_test")

colab_binary(
    name = "notebook_custom_kernel",
    exec_properties = {
        "mem": "24g",
    },
    g3_multiprocessing = True,
    paropts = ["--compress"],  # go/par-nocompress
    tags = ["optonly"],
    visibility = ["//third_party/py/orbax/docs:__subpackages__"],
    deps = [
        "//learning/brain/research/jax:tpu_support",
        "//third_party/py/absl/flags",
        "//third_party/py/flax",
        "//third_party/py/optax",
        "//third_party/py/orbax/experimental/model/jax2obm",
        "//third_party/py/orbax/experimental/model/tf2obm",
        "//third_party/py/orbax/export",
        "//third_party/py/tensorflow_datasets",
        "@jax",
        "@neptune//runtime/python:orbax_model_runner",
        "@numpy",
        "@orbax//:checkpoint",
        "@orbax//:experimental_model_core",
    ],
)

colab_notebook_test(
    name = "mnist_train_and_inference_test",
    colab_binary = ":notebook_custom_kernel",
    default_cell_diff = "ignore",
    ipynb = "mnist_train_and_inference.ipynb",
    tags = [
        "requires-dragonfish:4",
    ],
)

ipynb_deps(
    name = "mnist_train_and_inference_deps",
    srcs = ["mnist_train_and_inference.ipynb"],
    pytype = False,
    deps = [
        "//third_party/py/IPython:ipython-libs",
        "//third_party/py/PIL:pil",
        "//third_party/py/absl/flags",
        "//third_party/py/etils/epath",
        "//third_party/py/flax:core",
        "//third_party/py/flax/metrics:tensorboard",
        "//third_party/py/flax/training:train_state",
        "//third_party/py/frozendict",
        "//third_party/py/matplotlib",
        "//third_party/py/ml_collections",
        "//third_party/py/optax",
        "//third_party/py/orbax/export:constants",
        "//third_party/py/orbax/export:export_manager",
        "//third_party/py/orbax/export:jax_module",
        "//third_party/py/orbax/export:obm_configs",
        "//third_party/py/orbax/export:serving_config",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
        "//third_party/py/tensorflow_datasets",
        "@com_google_absl_py//absl/logging",
        "@jax",
        "@jax//:experimental_mesh_utils",
        "@neptune//runtime/python:orbax_model_runner",
        "@numpy",
        "@orbax//:checkpoint",
    ],
)
