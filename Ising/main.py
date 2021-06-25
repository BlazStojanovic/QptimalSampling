"""
Main file for running the Ising model example. 

Adapted from Flax Examples at ~ https://github.com/google/flax/blob/65061e6128f6695eed441acf2bfffc3b1badd318/examples/mnist/main.py
"""

from absl import app 
from absl import flags
from absl import logging 

from clu import platform 
import train
import jax
from ml_collections import config_flags
import tensorflow as tf


FLAGS = flags.FLAGS

flags.DEFINE_string('workdir', None, 'Directory to store model data')
config_flags.DEFINE_config_file('config', None, 'File path to the hyerparam config', lock_config=True)


def main(argv):
	if len(argv) > 1:
		raise app.UsageError('Too many command-line arguments.')

	# We need to hide GPUs from TensorFlow, so they are available for JAX
	tf.config.experimental.set_visible_devices([], 'GPU')

	logging.info('JAX host: %d / %d', jax.process_index(), jax.process_count())
	logging.info('JAX local devices: %r', jax.local_devices())

	# Add a note so that we can tell which task is which JAX host.
	# (Depending on the platform task 0 is not guaranteed to be host 0)
	platform.work_unit().set_task_status(f'host_id: {jax.process_index()}, host_count: {jax.process_count()}')
	platform.work_unit().create_artifact(platform.ArtifactType.DIRECTORY,FLAGS.workdir, 'workdir')

	optimised_rates = train.train_and_evaluate(FLAGS.config, FLAGS.workdir)

	# TODO serialise the model and store for further use
	print("TODO: you need to serialize the model and store it for further samling")


if __name__ == '__main__':
	flags.mark_flags_as_required(['config', 'workdir'])
	app.run(main)