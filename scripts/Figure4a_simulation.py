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
                             ObjectCutting,
                             ChangedStartValue, PropertyDependence)
from cytostoch import analyzer
import math
import cmath
from matplotlib import pyplot as plt

max_x = 30
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
stable_growing_state = State(initial_condition=0, name="stable_growing")
stable_pausing_state = State(initial_condition=0, name="stable_pausing")

# ---FUNCTIONS TO MAP GROWING SHRINKING MODEL TO GROWING PAUSING MODEL -------
def get_gupu(lifetime, v_shrink_val, v_g_val, k_PuGu_val):

    p = (-2 * v_shrink_val * k_PuGu_val / v_g_val - 
         v_shrink_val / (lifetime * v_g_val))
    q = (- v_shrink_val / (lifetime * v_g_val) * k_PuGu_val + 
         (v_shrink_val**2 / v_g_val**2) * k_PuGu_val**2)
    
    k_GuPu_calculated = - p/2 - np.sqrt((p/2)**2 - q)
    k_GuPu_val = k_GuPu_calculated
    
    # check whether half life with calculated rescue rate is as expected
    H = (v_g_val * v_shrink_val * (k_GuPu_val + k_PuGu_val) / 
         ((- v_g_val * k_GuPu_val + v_shrink_val*k_PuGu_val) * 
          ( k_PuGu_val * v_shrink_val - k_GuPu_val * v_g_val)))
    assert round(H, 3) == round(lifetime, 3)
    
    rate = k_GuPu_val / v_shrink_val - k_PuGu_val / v_g_val
    print("Expected length: ", - 1/rate)
    print("Calculated rescue: ", k_GuPu_calculated)
    print("Target half life: ", lifetime)
    
    return k_GuPu_val


def get_vd_kcu(lifetime, k_GuPu_val, k_PsPu_val, k_PuGu_val, 
               v_g_val, v_shrink_val):
    # average length of growth-shrinkage model is -1/rate (calculated above)
    # now use equation for average length of pause model to calculate v_g
    # so that the average length is the same as in
    # growth-shrinkage model
    
    # calculate uncorrected k_cu from half_life
    k_cu = 1/lifetime
    
    # correct k_cu due to MTs being in pausing and growing state
    rate = k_GuPu_val / v_shrink_val - k_PuGu_val / v_g_val
    k_cu = (((k_GuPu_val + k_PsPu_val) / k_PuGu_val * k_cu + k_cu) / 
            (1 - 1/k_PuGu_val))
    
    v_g_calculated = (- (k_PuGu_val * k_GuPu_val / 
                         (k_cu + k_GuPu_val + k_PsPu_val) 
                         - k_PuGu_val) * (-1/rate))
    print("Calculated v_g: ", v_g_calculated)
    print("Adjusted k_cu: ", k_cu)
    # make sure that the length from the growth shrink model is the same as 
    # the length from the growth pause model
    length_exp = (- 1/((k_PuGu_val * k_GuPu_val / 
                        (k_cu + k_GuPu_val + k_PsPu_val) 
                        - k_PuGu_val)/v_g_calculated))
    assert round(length_exp,3) == round(- 1/rate,3) 

    return v_g_calculated, k_cu

# -------------------- DEFINE PARAMETER VALUES ------------------------------
base_v_f_val = 0.5
v_f_val = 0.5

v_f = Parameter(values = [0.5, 0.2
                          ], name="v_f")

k_PuGu_val = 10.8
v_g_val = 17
v_shrink_val = 7

k_PsPu_val = 0.04

lifetime = 1 / 0.5
k_GuPu_val = get_gupu(lifetime, v_shrink_val, v_g_val, k_PuGu_val)

v_d, k_cu_val =  get_vd_kcu(lifetime, k_GuPu_val, k_PsPu_val, k_PuGu_val, 
                        v_g_val, v_shrink_val)

v_g = Parameter(values = [v_d], name="v_g")

k_N = Parameter(values = np.array([
                                    0.2,
                                    ]) * (max_x - min_x), name="k_N")

k_NS = Parameter(values = np.array([5.5,
                                    ]), name="k_NS")


k_N_MTs = Parameter(values = np.array([
                                    1.7,
                                    ]), name="k_N_MTs")


k_N_MT_resources = Parameter(values=[
                                    2.8 * max_x,
                                    ], name="k_N_MT_resources")


k_PuGu = Parameter(values = [k_PuGu_val], name="k_PuGu") # measured


k_GuPu = Parameter(values = [
                            k_GuPu_val, 
                             ], name="k_GuPu")#0.5



k_cu_baseline = Parameter(values = [
                                    k_cu_val*2,
                                    ], name="k_cu_baseline")
k_cu_change = Parameter(values = [
                                -k_cu_val*2/2, 
                                  ], name="k_cu_change")
k_cu_tip = PropertyDependence(start_val=None, end_val=k_cu_baseline,
                                      param_change=k_cu_change,
                                      properties=[position,
                                                  stable_length,
                                                  unstable_length,
                                                  ],
                                      param_change_is_abs=True,
                                      prop_change_is_abs=True)

k_cu = Parameter(values = [
                              k_cu_val,
                           ], name="k_cu",
                    dependence=k_cu_tip
                 )

base_PsPu = k_PsPu_val
                                      
k_PsPu = Parameter(values = [
                            base_PsPu,
                              ],
                    name="k_PsPu")

k_cs_target = 0.04
k_Gs_val = 0.5
Psu_factor = k_Gs_val /(k_PsPu_val + k_cu_val)
k_cs_factor = (1 + Psu_factor + (k_PuGu_val/k_Gs_val *
                                ((k_PsPu_val + k_cu_val)/(k_PuGu_val*k_Gs_val) + 
                                 k_GuPu_val/k_PuGu_val) 
                            - k_GuPu_val/k_Gs_val) * Psu_factor)

k_cs_val = k_cs_target * k_cs_factor



k_cs = Parameter(values = [
    
                        k_cs_val
                        ],
                 name="k_cs")


k_GsPs = Parameter(values = [
                              k_Gs_val,#2,2
                             ], name="k_GsPs")



MTRF = Action(position, operation="subtract", parameter = v_f, name="MT-RF",
              )

growth = Action(unstable_length, operation="add",parameter = v_g, 
                states=[labile_growing_state],
                name="MT_growth")

stable_growth = Action(unstable_length, operation="add",parameter = v_g, 
                states=[stable_growing_state],
                name="MT_growth")

out_of_neurite = ObjectRemovalCondition([unstable_length, 
                                         stable_length, 
                                         position], 
                                        combine_properties_operation="sum",
                                        compare_to_threshold="smaller",
                                        threshold=-5)

nucleation = ObjectCreation(state=labile_growing_state, 
                             parameter = k_N , name="nucleation",
                              track_creation_sources=True,
                             creation_on_objects=False)

at_position_zero = ChangedStartValue(object_property=position,
                                     new_start_values=0)
nucleation_at_zero = ObjectCreation(state=labile_growing_state, 
                             parameter = k_NS , name="nucleation in soma",
                             changed_start_values=[at_position_zero])


nucleation_on_MTs = ObjectCreation(state=labile_growing_state, 
                             parameter = k_N_MTs , name="nucleation on MTs",
                                properties_for_creation=[
                                unstable_length,
                                    stable_length, 
                                    ],
                             resources = k_N_MT_resources,
                             creation_on_objects=True)


# -------------------- DEFINE STATE TRANSITIONS ------------------------------
labile_growing_to_pausing = StateTransition(start_state=labile_growing_state,
                                            end_state=labile_pausing_state,
                                            parameter = k_PuGu,
                                            name="labile_growing_to_pausing")

labile_pausing_to_growing = StateTransition(start_state=labile_pausing_state,
                                            end_state=labile_growing_state,
                                            parameter = k_GuPu,
                                            name="labile_pausing_to_growing")

labile_catastrophe_pausing = StateTransition(start_state=labile_pausing_state,
                                     parameter = k_cu,
                                     name="labile_catastrophe_pausing")

labile_pausing_to_stable = StateTransition(start_state=labile_pausing_state,
                                           end_state=stable_state,
                                           transfer_property=[unstable_length,
                                                              stable_length],
                                           parameter = k_PsPu,
                                           name="labile_pausing_to_stable")

stable_catastrophe = StateTransition(start_state=stable_state,
                                     parameter = k_cs,
                                     name="stable_catastrophe")

labile_growth_on_stable = StateTransition(start_state=stable_state,
                                          end_state=stable_growing_state,
                                          parameter = k_GsPs,
                                          name="labile_growth_on_stable")

stable_growing_to_pausing = StateTransition(start_state=stable_growing_state,
                                           end_state=stable_pausing_state,
                                           parameter = k_PuGu,
                                           name="stable_growing_to_pausing")

stable_pausing_to_growing = StateTransition(start_state=stable_pausing_state,
                                            end_state = stable_growing_state,
                                            parameter = k_GuPu,#k_GuPu
                                            name="stable_pausing_to_growing")

labile_catastrophe_to_stable_pausing = StateTransition(start_state=stable_pausing_state,
                                                        end_state=stable_state,
                                                        properties_set_to_zero=
                                                        [unstable_length],
                                                        parameter = k_cu,
                                                        name="labile_catastrophe_to_stable_pausing")

stable_pausing_to_stable = StateTransition(start_state=stable_pausing_state,
                                           end_state=stable_state,
                                           transfer_property=[unstable_length,
                                                              stable_length],
                                           parameter = k_PsPu,
                                           name="stable_pausing_to_stable")


# -------------------- DEFINE DATA EXTRACTIONS ------------------------------
neurite = Dimension(positions=[position], lengths=[stable_length])

local_stable_density_extractor_sum = DataExtraction([neurite],
                                        operation="2D_to_1D_density",
                                        resolution=0.2, 
                                        show_data=True,
                                        state_groups={"all":[labile_growing_state, 
                                                              labile_pausing_state,
                                                              stable_state,
                                                               stable_growing_state,
                                                               stable_pausing_state
                                                              ]})

local_unstable_density_extractor_sum = DataExtraction([neurite],
                                        operation="2D_to_1D_density",
                                        resolution=0.2,
                                        show_data=True,
                                        state_groups={"all":[labile_growing_state, 
                                                              labile_pausing_state,
                                                              stable_state,
                                                               stable_growing_state,
                                                               stable_pausing_state
                                                              ],
                                                      },
                                        )

global_stat_extractor = DataExtraction([neurite], 
                                        operation="global",
                                        properties={
                                                    "length":[unstable_length, 
                                                              stable_length]},
                                        state_groups="all",
                                        show_data=True,
                                        print_regularly=True)

local_density_extractor_all = DataExtraction([neurite],
                                        operation="2D_to_1D_density",
                                        resolution=0.2,
                                        show_data=False,
                                        state_groups={"all":[labile_growing_state, 
                                                              labile_pausing_state,
                                                              stable_state,
                                                               stable_growing_state,
                                                               stable_pausing_state
                                                              ]})
    
    
simulations_summary_path = "C:\\Users\\schelskim\\Nextcloud\\01ANALYSIS\\TUBB\\simulations\\"

MT_simulation = simulation.SSA(states=[labile_growing_state, 
                                       labile_pausing_state,
                                       stable_state, 
                                        stable_growing_state,
                                        stable_pausing_state
                                        ],
                                transitions=[
                                            nucleation,
                                            nucleation_at_zero, 
                                            nucleation_on_MTs,
                                            
                                            labile_growing_to_pausing, 
                                            labile_pausing_to_growing, 
                                            labile_catastrophe_pausing,
                                            labile_pausing_to_stable,
                                            stable_catastrophe,
                                             
                                            labile_growth_on_stable,
                                            stable_growing_to_pausing,
                                            stable_pausing_to_growing,
                                            labile_catastrophe_to_stable_pausing,
                                            stable_pausing_to_stable,                                            
                                            ], 
                               properties=[position,
                                           stable_length,
                                           unstable_length],
                               actions=[MTRF, growth, stable_growth],
                                object_removal=[out_of_neurite],
                               device="gpu",
                               script_path=__file__,
                               simulations_summary_path=
                               simulations_summary_path)

comment = "New fit (corrected vals)"

nb_simulations = 2000
min_t = 450
torch.manual_seed(42)

experiment = "Fig4a_simulation"
folder = # REPLACE THIS WITH THE FOLDER WHERE YOU WANT TO SAVE THE SIMULATION OUTPUT
data_folder = os.path.join(folder, experiment)

MT_simulation.start(nb_simulations, min_t, 
                    data_extractions={
                                    "global":global_stat_extractor,
                                    "local_density_all":local_density_extractor_all,
                                    "local_stable_density":
                                        local_stable_density_extractor_sum,
                                    "local_unstable_density":
                                        local_unstable_density_extractor_sum,
                                          },
                    data_folder=data_folder,
                    nb_parallel_cores=32,
                    max_number_objects=2000,
                    all_parameter_combinations=False,
                    single_parameter_changes=False,
                    always_add_number_to_path=True,
                    time_resolution = 0.5,
                    start_save_time = min_t,
                    track_object_creation_time=True,
                    track_local_object_lifetime=False,
                    save_results=True,
                    print_times=True,
                    show_data=True,
                    gpu_number=0,
                    local_resolution=1,
                    local_lifetime_resolution=4,
                    comment=comment,
                    seed=42,
                    )


analysis = analyzer.Analyzer(simulation=MT_simulation)
analysis.start(time_resolution=min_t, max_time=min_t)

