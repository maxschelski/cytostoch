# CytoStoch
Stochastic simulation (SSA) of microtubules in neurites, implemented using jit from numba (numba.cuda.jit) for fast parallel processing on a GPU. 
Programmed in a generalized fashion, to be used for stochastically simulating any model with different species, with optional properties for each species member, in an arbitrary spatial domain. 

The example under \scripts contains a simple simulation of MTs in the neurite running on a GPU.

In order to run scripts, an NVIDIA GPU must be available at the machine.

Note: Execution on CPUs currently not supported. Will be added in the future.

# Installation

The package was developed and tested in Windows.
<br/>Installation via Mamba (using the conda-forge channel) is recommended.

1. If you don't already have Mamba installed: Download and install Mamba from https://github.com/conda-forge/miniforge.
2. If you don't already have git installed: Download and install git from https://git-scm.com/downloads
3. Open an Anaconda terminal, navigate to the folder where you want to put cytotorch and clone the cytostoch repository:
> git clone https://github.com/maxschelski/cytostoch.git
4. Navigate into the folder of the repository (cytostoch):
> cd cytostoch
5. Create environment for cytostoch with Anaconda:
> mamba env create -f environment.yml
6. Activate environment in anaconda:
> conda activate cytostoch
6. Install cytostoch locally using pip:
> pip install -e .
7. Optional: If needed, install spyder to create, edit and run scripts:
> mamba install spyder
8. Optional: start spyder
> spyder
9. You can now import cytotorch to build and simulate your model:
> import cytostoch

The installation should take several minutes, if mamba and git have already been installed.

# Demo

A script to run simulations corresponding to Fig. 4a is included in the \script folder ("Figure4a_simulation.py")
In line 407 you should enter an absolute path for where you want the simulation data to be saved.
<br/>Cytostoch will create the folder "Fig4a_simulation" at your defined path and will save all the output files in there.
<br/>Cytostoch also saves different meta data including a pickled simulation object.
<br />The defined data extractions will extract the following data from the endpoint of the simulation - all saved as "data.feather" in the respective folders:
1. Global statistics (e.g. number of MTs) in folder "global"
2. Density across simulated neurite from all states summed in the folder "local_density_all"
3. Density of stable MTs across simulated neurite from all states summed in the folder "local_stable_density"
4. Density of unstable MTs across simulated neurite from all states summed in the folder "local_unstable_density"
<br/>Using an NVIDIA RTX4090 GPU the run time should be around 9 min.
