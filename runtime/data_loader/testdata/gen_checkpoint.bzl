"""Custom rules for Checkpoint testdata generation.

Blaze custom rule that runs an executable and exports output files under a
directory named after the target name. Implemented as a custom rule since
`native.genrule` doesn't support generating directories with arbitrary files.
"""

def _gen_checkpoint_impl(ctx):
    output_directory = ctx.actions.declare_directory(ctx.attr.name)
    outputs = [output_directory]
    args = ctx.actions.args()
    args.add("--output_dir={}".format(output_directory.path))
    args.add_all(ctx.attr.extra_args)
    ctx.actions.run(
        mnemonic = "GenCheckpoint",
        outputs = outputs,
        arguments = [args],
        env = ctx.attr.env,
        progress_message = (
            "Generating Checkpoint to {} ".format(
                output_directory.path,
            )
        ),
        executable = ctx.executable.exec_tool,
    )

    return [DefaultInfo(files = depset(outputs))]

_gen_checkpoint = rule(
    implementation = _gen_checkpoint_impl,
    doc = """Generates Checkpoint.

    `arguments` are passed as is to `exec_tool`.
    Additional flags can be passed by providing `extra_args`.
    """,
    attrs = {
        "exec_tool": attr.label(
            executable = True,
            cfg = "exec",
            mandatory = True,
        ),
        "extra_args": attr.string_list(),
        "env": attr.string_dict(),
    },
)

def gen_checkpoint(name, exec_tool, args = [], env = None):
    """Generates Checkpoint to the `{name}_out` directory.

    Args:
      name: Target name
      exec_tool: The executable target to run.
      args: Arguments to be passed to the generator binary.
      env: Environment variables to be set when running the generator binary.
    """

    # Rules that run the binary.
    _gen_checkpoint(
        name = name + "_gen",
        testonly = True,
        exec_tool = exec_tool,
        extra_args = args,
        env = env,
    )

    # Create a single target that contains generated files.
    native.filegroup(
        name = name,
        testonly = True,
        srcs = [name + "_gen"],
    )
