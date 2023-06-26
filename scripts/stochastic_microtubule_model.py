# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 11:00:41 2023

@author: Maxsc
"""
import numpy as np
import torch
from cytotorch import simulation
from cytotorch.basic import (ObjectProperty, PropertyGeometry, Action, State, 
                             ObjectRemovalCondition, StateTransition, 
                             Dimension, DataExtraction)
from cytotorch import analyzer
import time

max_x = 20
position = ObjectProperty(max_value=max_x, start_value=[0, max_x])
length_geometry = PropertyGeometry([position], 
                                   operation="same_dimension_forward")
length = ObjectProperty(max_value=length_geometry, start_value=0)

growing_state = State(initial_condition=50)

growth = Action(length, operation="add",values=15)
MTRF = Action(position, operation="subtract", values=np.linspace(0,2,5))

out_of_neurite = ObjectRemovalCondition([length, position], "sum_smaller_than",
                                        threshold=0)

def nucleation_time(time_array, frequency=0.2, min_nucleation_fraction=0.5):
    """
    Get nucleation rate at defined time
    """
    # 1 has to be added to sin function to prevent negative nucleation
    # then normalize nucleation to be within min_nucleation_fraction and 1
    return ((torch.sin(time_array*frequency) + 1) / (2 / (1 - min_nucleation_fraction))
            + min_nucleation_fraction)

max_nucleation_rates = np.linspace(1, 10, 5)
nucleation = StateTransition(end_state=growing_state, 
                             rates=max_nucleation_rates,
                              time_dependency=None)

MT_lifetimes = np.linspace(1, 60, 5)
catastrophe = StateTransition(start_state=growing_state,
                              lifetimes=MT_lifetimes)

neurite = Dimension(position=position, length=length)

data_extractor = DataExtraction([neurite], operation="2D_to_1D_density",
                                resolution=0.5)

MT_simulation = simulation.SSA(states=[growing_state],
                               transitions=[nucleation, catastrophe],
                               properties=[position, length],
                               actions=[growth, MTRF],
                               object_removal=out_of_neurite)

nb_simulations = 100
min_t = 60 
# folder = "C:\\Users\\Maxsc\\Desktop\\simulation"
folder = "E:\\Max\\simulation\\05_200_60"
# folder = "C:\\Users\\Maxsc\\Desktop\\sim_test"
start = time.time()
MT_simulation.start(nb_simulations, min_t, data_extraction=data_extractor,
                    data_folder=folder, print_update_time_step=0.5)

# MT_simulation.save(time_resolution=0.5, max_time = 60, 
#                    data_folder=folder)

analysis = analyzer.Analyzer(data_folder=folder)
analysis.start(time_resolution=0.5, max_time=min_t)
print("TIME: ", time.time() - start)
