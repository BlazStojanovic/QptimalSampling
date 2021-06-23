"""
Convolutional Neural Networks for approximating the rates of the model

"""


import jax
import jax.np as jnp
import jax.experimental.stax
from jax.experimental.stax import (BatchNorm, Conv, Dense, Flatten, Relu, LogSoftmax)


