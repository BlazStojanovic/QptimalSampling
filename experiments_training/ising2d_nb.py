import sys
sys.path.append('../')

import Trainer as tr
import Sampler as sa
import jax.profiler

import configs.ising_defaults as iconf

if __name__ == '__main__':
	# server = jax.profiler.start_server(9999)
	jax.profiler.start_trace("/tmp/tensorboard")

	for nbc in [10]:
		# IO details
		conf = iconf.get_defaults()
		conf.batch_size = nbc
		out = "nb{}l20".format(int(nbc))

		# construct experiment
		ising_ex1 = tr.Trainer(experiment_name="ising_2d_meeting1_fig",
				 config=conf,
				 output_dir=out)

		ising_ex1.setup_experiment()
		out = ising_ex1.train_rates()

	# out.block_until_ready()

	jax.profiler.save_device_memory_profile("memory.prof")
	jax.profiler.stop_trace()