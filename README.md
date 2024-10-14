This is the repository for the paper: "Adapting to time why nature may have evolved a diverse set of neurons"

Paper url:

This code uses the conda environment: NeuromorphEnv


## Notes:
The simulations were performed using 4 3090 GPUs, it is recommended to use such a setting. However, the code can also run with 1 GPU.

All the simulations can take several weeks, so kindly have have this in mind when running them.

## Instructions:
### Main studies
  -Go to Main and then go to the psettings file \
  -In the psettings one can choose which experiment to run by setting its flag to True (and others to False) in the EXP_FLAGS variable \
  -The settings for each simulations is in its designated "get_exp_number", for example the first experiment is "get_exp_one" \
  -The affected plots in the paper are in the docstring of the experiment \
  -The meaning of each simuation parameter is in the get_exp_one setting
### Noise studies
  -These are the simulations for Fig.5 in the paper \
  -Go to main.py and choose which simulation to perform \
  -The associated parameters are listed under
