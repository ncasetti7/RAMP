# Reaction Analysis with Machine-Learned Potentials - RAMP

## Installation Instructions
To install RAMP, the first step is to clone the repo
```
git clone <INSERT_LINK>
```
Next, create and activate a conda environment 
```
cd RAMP
conda env create --file environment.yml
conda activate RAMP
```
Then pip install the repo
```
pip install -e .
```
(Optional) Run the unit tests to ensure the implementation was succesful.
This currently won't work because AIMNet2-rxn isn't currenltly available!
```
cd test
python -m unittest
```

## How To Perform a Mechanism Search
This tutorial will walk through running RAMP on an example sytem and then explain how to run custom searches by modifying the config file. Important note: The reactive potential used to generate the results from the paper is currently not publicly available!
### Run RAMP up until QM calculations
To run RAMP up until QM calculations are performed, change into the run_scripts directory and run the run_from_config.py script
```
cd run_scripts
python run_from_config.py ../config/example_config/search_01.yaml
```
This will run RAMP on the example system
### Run QM Calculations with RAMP
To run the QM calculations associated with RAMP, a slurm scheduler is required. To run the calculations, change into the run_scripts directory as before and run the run_qm_from_config.py script
```
cd run_scripts
python run_qm_from_config.py ../config/example_config/search_01.yaml
```
This will run RAMP's QM calculations on the example system. This won't work until the initial RAMP run is performed.
### Visualize results and setup next search
To visualize results, open the visualize_network.ipynb notebook in results and input your canonical reactant, config folder name, and search number. This will allow you to visualize the network as it stands and will also automatically select and setup a config file for the next molecule for investigation.
### Config Files
To run a custom mechanism search, the first step is to make a folder to house the config files (see an example in config/example_config). The naming convention for config files is config_## where ## indicates how many searches have been performed for this system (the first search is named search_01). Many of the default values in the config file are usable for most searches, however, there are a few that are important to adjust to ensure proper function
```
with_slurm: bool # Whether to submit the job as a slurm script (slurm is the only queueing system currently supported)
path_to_mechanism_search: /PATH/TO/RAMP # File path to RAMP
template_slurm_script: src/RAMP/utils/template_slurm_script.sh # Used if with_slurm is True, how RAMP submits jobs see the default for an example
input_structure: SMILES # Reactant of interest, needs to be canonical!
barrier_model_file: /PATH/TO/MODEL # Path to NNP model (must be a .jpt AIMNet model)
orca_path: /PATH/TO/ORCA # If running QM calculations, orca will be used (slurm scheduler necessary to run QM calculations currently)
```
