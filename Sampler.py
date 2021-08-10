"""
Sampler class, a class for sampling from the parameterised rates
"""

import os
import sys
from dataclasses import dataclass
import ml_collections

from qsampling_utils.sampl_utils import step_max, step_gumbel
from qsampling_utils import pCNN1d, pCNN2d


@dataclass
class Sampler:
	pass