"""Constants for the Neptune Model."""

################################################################################
# SHLO constants
################################################################################

SHLO_MIME_TYPE = 'application/x.mlir-stablehlo'
SHLO_VERSION = '1.0'

################################################################################
# Manifest model format constants
################################################################################

# The filename of the model version metadata file relative to the save
# directory.
MODEL_VERSION_FILENAME = 'neptune_model_version.txt'

# The file path of the manifest proto file relative to the save directory.
MANIFEST_FILE_PATH = 'manifest.pb'

# The version of the manifest.
MANIFEST_VERSION = '0.0.1'

# The mime type of the manifest proto file.
MANIFEST_MIME_TYPE = (
    'application/protobuf; type=neptune_model_manifest.Manifest'
)
