from setuptools import find_packages, setup

setup(
	name="cytotorch",
	author="Max Schelski",
	author_email = "max.schelski@googlemail.com",
	description = "Stochastic simulations of cytoskeletal structures.",
	long_description = " Intended for microtubules in neurites, but build to be much more flexible.",
	version="0.0.2",
	keywords = ["python", "pytorch", "stochast simulation", "SSA", "GPU", "Gillespie", "cytoskeleton", "spatial modeling"],
	packages=find_packages()
      )