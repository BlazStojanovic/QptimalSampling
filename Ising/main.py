"""
Main file for running the Ising model example. 
"""

from absl import app 
from absl import flags
from absl import logging 

from clu import platform 
import train
import jax
from ml_collections import config_flags
import tensorflow as tf

