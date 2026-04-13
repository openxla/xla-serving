"""Constants for Neptune Export."""

ORCHESTRATION_SUPPLEMENTAL_NAME: str = "orchestration"

DEFAULT_ORCHESTRATION_SUPPLEMENTAL_FILENAME: str = "orchestration.pb"

DEFAULT_CHECKPOINT_SUBDIR: str = "checkpoint"

WEIGHTS_NAME: str = "weights"

ORBAX_CHECKPOINT_MIME_TYPE: str = "application/x.orbax-checkpoint"


################################################################################
# Logging Messages
################################################################################
# Prefix to use for prepending to logging messages
LOG_PREFIX: str = "[NeptuneExport]"
