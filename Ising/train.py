"""
Utilities for training the Ising model

"""

import sys
sys.path.append('../')

from qsampling_utils.loss import ising_endpoint_loss

from absl import logging
from flax import linen as nn
from flax.metrics import tensorboard
from flax.training import train_state

import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax

from functools import partial
from typing import Any, Callable, Sequence, Tuple

ModuleDef = Any

loss = ising_endpoint_loss