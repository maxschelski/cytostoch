from setuptools import find_packages, setup

setup(
	name="cytostoch",
	author="Max Schelski",
	author_email = "max.schelski@googlemail.com",
	description = "Stochastic simulations of cytoskeletal structures.",
	long_description = " Intended for microtubules in neurites, but build to be much more flexible.",
	version="0.1.0",
	keywords = ["python", "pytorch", "numba", "cuda", "stochastic simulation",
				"SSA", "GPU", "Gillespie", "cytoskeleton", "spatial modeling"],
	packages=find_packages()
      )