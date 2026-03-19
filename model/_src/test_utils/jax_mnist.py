"""JAX MNIST model for testing."""

import flax.linen as nn


class JaxMnist(nn.Module):
  """Flax MNIST model."""

  @nn.compact
  def __call__(self, x):
    """See base class."""
    x = nn.Conv(features=32, kernel_size=(4, 4))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = nn.Conv(features=64, kernel_size=(4, 4))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = x.reshape((x.shape[0], -1))  # flatten
    x = nn.Dense(features=256)(x)
    x = nn.relu(x)
    x = nn.Dense(features=10)(x)
    x = nn.log_softmax(x)
    return x
