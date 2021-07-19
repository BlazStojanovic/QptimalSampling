"""
Trainer class, a class for training and conducting training experiments for learning the rates 
for lattice models
"""


import sys
from dataclasses import dataclass

# qsampling utils imports 
from qsampling_utils.sampler import step_max, step_gumbel
from qsampling_utils.pCNN import pCNN, CircularConv, check_pcnn_validity
from ising_loss import ising_endpoint_loss, ising_potential



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

		for epoch in range(1, setup.num_epochs+1):
			# split subkeys for shuffling purpuse
			key, subkey = rnd.split(key)

			# optimisation step on one batch
			self.state, self.vals = self.train_epoch()
			ll[epoch-1] = self.vals




if __name__ == '__main__':
	
	ex = Trainer(experiment_name="exp",
				 default_config="conf",
				 output_dir="dir")

