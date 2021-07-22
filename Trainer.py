"""
Trainer class, a class for training and conducting training experiments for learning the rates 
for lattice models
"""

import sys
from dataclasses import dataclass

# qsampling utils imports 
from qsampling_utils.sampler import step_max, step_gumbel
from qsampling_utils.pCNN import pCNN, CircularConv, check_pcnn_validity

# Lattice imports
from ising_loss import ising_endpoint_loss, ising_potential

# Jax imports
import jax.numpy as jnp
import jax.random as rnd


@dataclass
class Trainer:
	experiment_name: str
	default_config: str
	output_dir: str

	def train_rates(self, validate=True):
		"""
		train a model to find optimal rates for a lattice system
	
		Params:
		-------
		validate -- boolean if full validation is returned

		Returns:
		--------
		sampler -- Sampler object, which allows for importance sampling from the rates
		
		"""
		setup = self.simulation_params

		# PRNGKey is fixed for reproducibility
		key = rnd.PRNGKey(0)

		# variational approximation of the rates
		params, model = self.get_rate_parametrisation(key)

		# optimiser init
		tx = self.get_optimiser()

		# construct train state
		state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

		# loses
		ll = jnp.zeros((setup.num_epochs,))

		print("Solving for L = {}, J = {}, g = {}".format(config.lattice_size, config.J, config.g))
		print("T = {}, batch = {}".format(config.T, config.batch_size))
		print("-----------------------------------------------------------------------------------")

		for epoch in range(1, setup.num_epochs+1):
			# split subkeys for shuffling purpuse
			key, subkey = rnd.split(key)

			# optimisation step on one batch
			state, vals = train_epoch() # todo decide
			ll.at[epoch-1].set(vals) = vals


if __name__ == '__main__':
	
	ex = Trainer(experiment_name="exp",
				 default_config="conf",
				 output_dir="dir")

