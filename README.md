# CytoStoch
Stochastic simulation (SSA) of microtubules in neurites, implemented using jit from numba (numba.njit, numba.cuda.jit) for fast parallel processing on a GPU. 
Programmed in a generalized fashion, to be used for stochastically simulating any model with different species, with optional properties for each species member, in an arbitrary spatial domain. 

Example under scripts not compatible with current cytostoch version and therefore not running.

Note: Execution on CPUs currently not supported. Will be added again in the future.

# Installation

The package was developed and tested in Windows.
<br/>
1. If you don't already have Anaconda installed: Download and install Anaconda from https://www.anaconda.com/.
2. If you don't already have git installed: Download and install git from https://git-scm.com/downloads
3. Open a terminal, navigate to the folder where you want to put cytotorch and clone the cytostoch repository:
> git clone https://github.com/maxschelski/cytostoch.git
4. Navigate into the folder of the repository (cytostoch):
> cd cytostoch
5. Create environment for cytotorch with Anaconda:
> conda env create -f environment.yml
6. Activate environment in anaconda:
> conda activate cytostoch
6. Install cytostoch locally using pip:
> pip install -e .
7. You can now import cytotorch to build and simulate your model:
> import cytostoch
