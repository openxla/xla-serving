"""Constants used in jax2npt."""

# XLA compilation options.
XLA_COMPILE_OPTIONS = 'xla_compile_options'

# TODO: Consolidate constants related to mime types in a common library, so that both obm and jsv can use them.
# XLA compile options mime type.
XLA_COMPILE_OPTIONS_MIME_TYPE = (
    'application/protobuf; type=neptune_model_manifest.CompileOptionsProtoMap'
)
