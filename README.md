# QOptimal sampling source code - MPhil repository of Blaz Stojanovic

[![Hello, world](https://github.com/BlazStojanovic/QptimalSampling/blob/main/experiments_sampling/figures/rate_structure.png)]

## Testing the code

Tests for the implementation of the Ising loss and helper functions is in Ising/tests (analogous for Heisenberg), to run a test simply go into appropriate directory and run
```
python -m unittest test_something.py
```

## Reproducing the results
The structure of the repository may seem convoluted at first, but it follows a simple workflow. Each result/figure in the thesis has a corresponding run.py file, which stores the produced data in the data/ repository, if processing of the data is needed before plotting there exists a corresponding process.py file in the process/ directory, finally the plotting scripts can be found in the plot/ directory and they store the figures in the figures/ directory. To reproduce any figure in the thesis simply execute:

* target_result.py -> data/target/
* process/target_process.py 
* plot/target_plot.py -> figures/ 

this produces appropriately named figures. Runner files are found in experiments_training/ and experiments_sampling/ repositories, depending on type of result.

## Notes
If you have any questions, contact at blaz.sto@gmail.com