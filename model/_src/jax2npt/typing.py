from collections.abc import Callable
import jaxtyping


# ApplyFn take two arguments, the first one is the model_params, the second one
# is the model_inputs.
ApplyFn = Callable[[jaxtyping.PyTree, jaxtyping.PyTree], jaxtyping.PyTree]
