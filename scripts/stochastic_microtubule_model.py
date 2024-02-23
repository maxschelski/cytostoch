# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 11:00:41 2023

@author: Maxsc
"""
import os
import numpy as np
import torch
from cytostoch import simulation
from cytostoch.basic import (ObjectProperty, ObjectPosition,
                             PropertyGeometry, Parameter,
                             Action, StateTransition, State, 
                             ObjectRemovalCondition, 
                             Dimension, DataExtraction, ObjectCreation,
                             ChangedStartValue)
from cytostoch import analyzer

max_x = 20
min_x = 0
position = ObjectPosition(max_value=max_x, start_value=[min_x, max_x],
                          name="position")

stable_length = ObjectProperty(start_value=0,
                               initial_condition=0, name="stable_length")

unstable_length_geometry = PropertyGeometry([position, stable_length], 
                                    operation="same_dimension_forward")
unstable_length = ObjectProperty(max_value=unstable_length_geometry, 
                                 start_value=0, initial_condition=0, 
                                 name="unstable_length")

stable_length_geometry = PropertyGeometry([position, unstable_length], 
                                   operation="same_dimension_forward")
stable_length.max_value = stable_length_geometry

labile_growing_state = State(initial_condition=0, name="labile_growing")
labile_pausing_state = State(initial_condition=0, name="labile_pausing")
stable_state = State(initial_condition=0, name="stable")

v_f = Parameter(values = [0.5], name="v_f")
v_g = Parameter(values = [12], name="v_g")

k_N = Parameter(values = np.array([4]) * (max_x - min_x), name="k_N")

k_PuGu = Parameter(values = [7.5], name="k_PuGu")

k_GuPu = Parameter(values = [0], name="k_GuPu")

k_cu = Parameter(values = [1], name="k_cu")

k_PsPu = Parameter(values = [0.04], name="k_PsPu")

k_cs = Parameter(values = [0.035], name="k_cs")

MTRF = Action(position, operation="subtract", parameter = v_f, name="MT-RF")
growth = Action(unstable_length, operation="add",parameter = v_g, 
                states=[labile_growing_state],
                name="MT_growth")

out_of_neurite = ObjectRemovalCondition([unstable_length, 
                                         stable_length, 
                                         position], 
                                        combine_properties_operation="sum",
                                        compare_to_threshold="smaller",
                                        threshold=min_x)

nucleation = ObjectCreation(state=labile_growing_state, 
                             parameter = k_N , name="nucleation",
                             creation_on_objects=False)

labile_growing_to_pausing = StateTransition(start_state=labile_growing_state,
                                            end_state=labile_pausing_state,
                                            parameter = k_PuGu,
                                            name="labile_growing_to_pausing")

labile_pausing_to_growing = StateTransition(start_state=labile_pausing_state,
                                            end_state=labile_growing_state,
                                            parameter = k_GuPu,
                                            name="labile_pausing_to_growing")

labile_catastrophe = StateTransition(start_state=labile_pausing_state,
                                     parameter = k_cu,
                                     name="labile_catastrophe")

labile_pausing_to_stable = StateTransition(start_state=labile_pausing_state,
                                           end_state=stable_state,
                                           transfer_property=[unstable_length,
                                                              stable_length],
                                           parameter = k_PsPu,
                                           name="labile_pausing_to_stable")

stable_catastrophe = StateTransition(start_state=stable_state,
                                     parameter = k_cs,
                                     name="stable_catastrophe")


neurite = Dimension(positions=[position], lengths=[stable_length])
local_stable_density_extractor = DataExtraction([neurite],
                                        operation="2D_to_1D_density",
                                        resolution=0.02,
                                        state_groups="all")

neurite = Dimension(positions=[position], lengths=[unstable_length, 
                                                stable_length])
local_density_extractor = DataExtraction([neurite],
                                        operation="2D_to_1D_density",
                                        resolution=0.02,
                                        state_groups="all")

global_stat_extractor = DataExtraction([neurite], 
                                        operation="global",
                                        properties={"unstable_length":
                                                    [unstable_length],
                                                    "stable_length":[stable_length],
                                                    "length":[unstable_length, 
                                                              stable_length]},
                                        state_groups="all",
                                        print_regularly=True)
    
    
simulations_summary_path = "C:\\Users\\schelskim\\Nextcloud\\01ANALYSIS\\TUBB\\simulations\\"

MT_simulation = simulation.SSA(states=[labile_growing_state, 
                                       labile_pausing_state,
                                       stable_state
                                        ],
                                transitions=[
                                            nucleation,
                                            labile_growing_to_pausing, 
                                            labile_pausing_to_growing, 
                                            labile_catastrophe,
                                            labile_pausing_to_stable,
                                            stable_catastrophe,
                                            ], 
                               properties=[position, 
                                           unstable_length,
                                           stable_length],
                               actions=[MTRF, growth],
                                object_removal=[out_of_neurite],
                               device="gpu",
                               script_path=__file__,
                               simulations_summary_path=
                               simulations_summary_path)
    
nb_simulations = 500
min_t = 200
torch.manual_seed(42)

experiment = "MTRF_stable_MTs"

folder = "C:\\Users\\schelskim\\Nextcloud\\01ANALYSIS\\TUBB\\simulations\\MTRF_stochastic_steady_state\\"
data_folder = os.path.join(folder, experiment)

MT_simulation.start(nb_simulations, min_t, 
                    data_extractions={
                                    "global":global_stat_extractor,
                                      "local_density":local_density_extractor,
                                          "local_stable_density":
                                              local_stable_density_extractor
                                          },
                    data_folder=data_folder,
                    nb_parallel_cores=32,
                    max_number_objects=1000,
                    all_parameter_combinations=False,
                    single_parameter_changes=True,
                    time_resolution=min_t,
                    save_results=False,
                    )

MT_simulation.save(time_resolution=0.01, max_time = 0.1)

analysis = analyzer.Analyzer(simulation=MT_simulation, 
                              data_folder=data_folder)
analysis.start(time_resolution=min_t, max_time=min_t)
