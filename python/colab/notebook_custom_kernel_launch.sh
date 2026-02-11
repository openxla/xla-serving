#!/bin/bash

# Usage:
# bash third_party/neptune/python/colab/notebook_custom_kernel_launch.sh

source gbash.sh || exit

DEFINE_string xm_cell "viglobal" "XManager cell"
DEFINE_string tpu_platform "viperlite_pod=8" "TPU platform"

main() {
  XM_POOL="cml"
  XM_ALLOC="cml/cml-shared-ml-user"
  XM_CELL=${FLAGS_xm_cell}
  TPU_PLATFORM=${FLAGS_tpu_platform}
  NAME="${USER}-custom-kernel-${TPU_PLATFORM?}"

  blaze build -c opt --max_remote_input_files=100000 //third_party/neptune/python/colab:notebook_custom_kernel.par

  borg_vars="
  notebook_binary=./blaze-bin/third_party/neptune/python/colab/notebook_custom_kernel.par,
  cell=${XM_CELL?},
  ${TPU_PLATFORM},
  ram=128G,
  milligcu=1000,
  priority=200,
  kernel_args={tfrt_session_pathways_device_type='TPU'},
  registration_label=${NAME?}
  "
  gxm learning/deepmind/public/tools/ml_python/ml_notebook.borg \
      --vars="${borg_vars}" \
      --experiment_name="${NAME?}" \
      --xm_resource_pool="${XM_POOL?}" \
      --xm_resource_alloc="${XM_ALLOC?}" \
      --experiment_tags='idle_opt_out=colab' \
      --noxm_monitor_on_launch \
      --xm_experimental_no_par=false \
      --xm_use_persistent_pytype_workers=false
}

gbash::main "$@"
