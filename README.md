## RaceLineOptimisation_RL
Final Assignment of TU Delft's course AE4350

In this project, the race line of a given circuit is optimised for a Formula 1 car.

# List of available circuits:
* monaco
* monza
* portimao
* silverstone
* spa
* test curve
* zandvoort

# How to run
Using conda, create the environment using the `env.yml` file:
```console
conda env create -f env.yml
```

Run any code from the test folder, such as the `reward_test.py` file, to create a policy file (.d3 file). Feel free to change the general inputs from the `Inputs.py` file or the SAC inputs from `sac_inputs.yml`.

Then, run the `raceline.py` file to analyse the resulting learnt strategy.

# Note

The report can be found under the report folder.
