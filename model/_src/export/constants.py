"""Constants for Neptune Export."""

ORCHESTRATION_SUPPLEMENTAL_NAME: str = "orchestration"

ORCHESTRATION_MULTI_PIPELINE_MIME_TYPE: str = (
    "application/protobuf; type=neptune_orchestration.Pipelines"
)
ORCHESTRATION_MULTI_PIPELINE_VERSION: str = "0.0.1"

DEFAULT_SUPPLEMENTAL_FILENAME: str = "orchestration.pb"
