# -*- coding: utf-8 -*-

import torch
import copy
import numpy as np
import pandas as pd
import time
import shutil
import sys
import gc
import os
import psutil
import pickle
import dill

# NUMBA_DEBUG=1
# NUMBA_DEVELOPER_MODE = 1
# NUMBA_DEBUGINFO = 1
import numba
import numba.cuda.random
import math
from numba import cuda
from matplotlib import pyplot as plt
import seaborn as sns

from . import analyzer
from .basic import PropertyGeometry
from . import simulation_numba
import tqdm

"""
This is for single type modeling! (e.g. just MTs or actin) 
What to model?
- allow different action states of MTs and different state variables
- define transition rates between MT states
- define deterministic actions for states (grow, Pause, MT-RF);
    each action can be defined for multiple states
- allow different parameter value ranges
What classes?
- simulation class (executes the simulation)
- properties class (e.g. length and position)
- action class (defines what happens to a property, gets property supplied and
                what happens with it)
- state class for each state
- transition class (includes one transition, rates 
                                 (rate can be supplied as lifetime))
- action state (for changes to properties)
- Parameter class for each value of transition or action
"""

class tRSSA():

    def __init__(self):
        pass

    def start(self, time_step_size_simulation, time_depency_step_size=0.001):

        self.time_dependency_step_size = time_depency_step_size
        # create bounds (upper and lower) for objects

        # for time dependency get min and max points of time dependent function
        # first numerically simulate time dependent function over whole interval
        self.timesteps = torch.arange(0, self.max_time, self.time_depency_step_size)
        for transition in self.transitions:
            if transition.time_dependency is None:
                continue
            time_data = transition.time_dependency(self.timesteps)
            transition.time_depency_data = time_data

        for action in self.actions:
            if action.time_dependency is None:
                continue
            time_data = transition.time_dependency(self.timesteps)
            # multiplay by step size to get actual values and not rates/speeds
            time_data *= self.time_dependency_step_size
            action.time_dependency_data = time_data

        # get min and max for first time range
        index = torch.where((self.timesteps >= 0) &
                            (self.timesteps < time_step_size_simulation))

        # create bounds for propensities

        # draw random numbers with lambda = upper bound

        # check if random number is below lower bound

        # check for remaining positions, whether actual propensity at time
        # is higher than random number
        # for all where not the case, add one to counter

        # check whether object bounds hold, otherwise increase or decrease
        # bounds

    def run_iteration(self):
        pass


class SSA():

    def __init__(self, states, transitions, properties, actions,
                 object_removal=None, script_path=None,
                 simulations_summary_path = None,
                 name="", device="GPU",
                 use_free_gpu=False):
        """

        Args:
            states (list of State objects): The state for not present does not
                need to be defined
            transitions (list of StateTransition objects):
            properties (list of ObjectProperty objects): Order of object
                properties in list is important if a Geometry object is supplied
                for an object property (min, max, start or initial condition),
                then the object property that is a parameter to Geometry
                must be in the list before the property that contains the
                Geometry (e.g. the property "length" has a Geometry object based
                on the property "position", then the property "position" needs
                to be in the properties list before the property "length")
            actions (list of Action objects):
            name (str): Name of simulation, used for data export and readability
            object_removal (list of ObjectRemovalCondition objects): Define
                when an object should be removed from the simulation, based on
                its object properties
        """
        self.states = states
        self.transitions = transitions
        self.properties = properties
        self.actions = actions
        self.object_removal = object_removal
        self.script_path = script_path
        self.name = name
        self.simulations_summary_path = simulations_summary_path

        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

        torch.set_default_tensor_type(torch.FloatTensor)
        torch.set_default_device("cpu")
        # create tensors object to reference the correct tensor class
        # depending on the device
        if (device.lower() == "gpu") & (torch.cuda.device_count() > 0):
            if use_free_gpu:
                # use GPU that is not used already, in case of multiple GPUs
                for GPU_nb in range(torch.cuda.device_count()):
                    if torch.cuda.memory_reserved(GPU_nb) == 0:
                        break
                self.device = "cuda:"+str(GPU_nb)
            else:
                gpu_memory = []
                for GPU_nb in range(torch.cuda.device_count()):
                    device_props = torch.cuda.get_device_properties(GPU_nb)
                    free_memory = (device_props.total_memory
                                   - torch.cuda.memory_reserved(GPU_nb))
                    gpu_memory.append(free_memory)
                highest_free_memory_gpu = np.argmax(np.array(gpu_memory))
                self.device = "cuda:"+str(int(highest_free_memory_gpu))
            self.tensors = torch.cuda
            # torch.set_default_tensor_type(torch.cuda.FloatTensor)
            # torch.set_default_device(self.device)
        else:
            self.device = "cpu"
            self.tensors = torch
            torch.set_default_tensor_type(torch.FloatTensor)

        if (device.lower() == "gpu") & (len(numba.cuda.gpus) > 0):
            self.device = "cuda"

        # since state 0 is reserved for no state (no object), start
        for state_nb, state in enumerate(self.states):
            state.number = state_nb + 1
            state.name += "_" + str(state.number)

        for property_nb, property in enumerate(self.properties):
            property.number = property_nb

        for transition in self.transitions:
            if transition.name != "":
                continue
            start_state = transition.start_state
            if start_state is not None:
                start_state = start_state.number
            end_state = transition.end_state
            if end_state is not None:
                end_state = end_state.number
            transition.name = ("trans_" + str(start_state) +
                               "to" + str(end_state))

    def start(self, nb_simulations, min_time, data_extractions, data_folder,
              time_resolution, save_initial_state=False,
              max_number_objects=None,
               single_parameter_changes=True, nb_parallel_cores=1,
              all_parameter_combinations=False,
              seed=42,
              ignore_errors=False, print_update_time_step=1,
               nb_objects_added_per_step=10,
              max_iters_no_data_extraction=2000,
               dynamically_increase_nb_objects=False,
              remove_finished_sims=False, save_states=False,
              save_results=True, print_times=True,
              use_assertion_checks=True, reset_folder=True, bug_fixing=False,
              choose_gpu=False):
        """

        Args:
            nb_simulations (int): Number of simulations to run per parameter
                combination
            min_time (float): minimum time
            time_resolution ( float): time resolution of output data (in min)
            data_extractions (Dict of DataExtraction objects): Dict in which
                values are different DataExtraciton objects. Keys are names
                of extracted data.
            data_folder (string): Folder in which data should be saved
            max_number_objects (int): maximum number of objects allowed to be
                simulated. Determines array size
            all_parameter_combinations (Boolean): Whether the combination of all
                supplied parameter values from all parameters should be
                analyzed. This means that every value of every parameter is
                combined with every value of every other parameter and thereby
                simulations of all parameter combinations are executed.
            single_parameter_changes (Boolean): Whether parameter values should
                be used as changing one parameter value at a time from the
                standard value (first value). This way looking at changes of
                single parameters can be executed easily.
                If False, then each simulation will use parameter values from
                one index for all supplied parameters. Therefore, lists
                of values for all parameters have to have the same length -
                or the can have only one value, in which case this single value
                will be expanded to the total length of parameter values.
                This allows e.g. definition of pairs of parameter values to use
                (changing two parameters for a simulation) but also any other
                combination of parameter value changes.
        Returns:

        """
        # turn off autograd function to reduce memory overhead of pytorch
        # and reduce backend processes

        # select GPU
        if choose_gpu & (len(cuda.gpus) > 1):
            gpu_list = ("Choose one of all available GPUs by entering the "
                       "corresponding number and pressing ENTER:\n")
            gpu_names = []
            for number, gpu in enumerate(cuda.gpus):
                gpu_name = str(gpu._device.name.decode(encoding='utf-8',
                                                       errors='strict'))
                gpu_names.append(gpu_name)
                gpu_list += str(number)+": "+ gpu_name+"\n"

            enter_number = f"Enter an integer number from 0 and {number}: "
            gpu_list += enter_number
            print(gpu_list)
            valid_input = False
            while not valid_input:
                gpu_nb = input()
                try:
                    gpu_nb = int(gpu_nb)

                except:
                    print("You have to enter an integer number.\n"
                          + enter_number)
                    continue

                if (gpu_nb < 0):
                    print("You have to enter a non negative number.\n"
                          + enter_number)
                    continue

                if gpu_nb >= len(gpu_names):
                    print(f"You can't enter a number larger than {number}.\n"
                          + enter_number)
                    continue

                print(f"GPU {str(gpu_nb)} selected - "
                      f"{gpu_names[gpu_nb]}")
                valid_input = True
            cuda.select_device(gpu_nb)

        with torch.no_grad():
            self._start(nb_simulations, min_time, data_extractions, data_folder,
                        time_resolution, save_initial_state,
                        max_number_objects, all_parameter_combinations,
                        single_parameter_changes, nb_parallel_cores, seed,
                        ignore_errors=ignore_errors,
                        print_update_time_step=print_update_time_step,
                        nb_objects_added_per_step=nb_objects_added_per_step,
                        max_iters_no_data_extraction=max_iters_no_data_extraction,
                        dynamically_increase_nb_objects=
                        dynamically_increase_nb_objects,
                        remove_finished_sims=remove_finished_sims,
                        save_states=save_states,
                        save_results=save_results, print_times=print_times,
                        use_assertion_checks=use_assertion_checks,
                        reset_folder=reset_folder,
                        bug_fixing = bug_fixing)

    def save(self, time_resolution, max_time):
        analysis = analyzer.Analyzer(simulation=self)
        analysis.start(time_resolution, max_time,
                       use_assertion_checks=self.use_assertion_checks)

    def _regularly_print_data(self):
        # only print data if the current data was not yet printed
        # and if there is data buffer to be printed
        for keyword, data_extraction in self.data_extractions.items():
            if not data_extraction.print_regularly:
                continue
            data = data_extraction.extract(self, regular_print=True)
            for data_name, data_array in data.items():
                mean = torch.nanmean(data_array.to(torch.float))
                if (not np.isnan(mean.item())) & (mean.item() != 0):
                    print(data_name," : ",mean.item())

    def _get_property_start_val_arrays(self):
        # - all object properties over time
        nb_properties = len(self.properties)

        property_start_values = np.full((nb_properties, 2), math.nan)
        for property_nb, property in enumerate(self.properties):
            start_value = property.start_value
            if property.start_value is None:
                continue
            if type(property.start_value) in [float, int]:
                property_start_values[property_nb, 0] = start_value
            else:
                property_start_values[property_nb, :] = start_value
        return property_start_values

    def _get_property_vals_array(self):
        nb_properties = len(self.properties)
        properties_array = np.zeros((nb_properties,
                                     *self.properties[0].array.shape))
        for property_nb, property in enumerate(self.properties):
            properties_array[property_nb] = property.array.cpu()
        return properties_array

    def _get_property_extreme_val_arrays(self):
        nb_properties = len(self.properties)
        # if there is just one non-nan value in min_value
        # then this is the min_value
        # if there are multiple non-nan values in min_value
        # then these are property numbers, except
        # for the first value, which is threshold
        # and the second value, which is the operation
        # (-1 for subtracting property values,
        #  +1 for adding property values)
        # the same is true for the max_value
        max_nb_min_value_properties = 0
        max_nb_max_value_properties = 0
        for property in self.properties:
            if ((str(type(property._min_value)) ==
                 "<class 'cytostoch.basic.PropertyGeometry'>") &
                    (property.min_value is not None)):
                nb_min_value_properties = len(property._min_value.properties)
                max_nb_min_value_properties = max(max_nb_min_value_properties,
                                                  nb_min_value_properties)
            if ((str(type(property._max_value)) ==
                 "<class 'cytostoch.basic.PropertyGeometry'>") &
                    (property.max_value is not None)):
                nb_max_value_properties = len(property._max_value.properties)
                max_nb_max_value_properties = max(max_nb_max_value_properties,
                                                  nb_max_value_properties)
        # add two to the maximum size since two more entries are added for
        # the additional options (threshold and operation)
        if max_nb_max_value_properties > 0:
            max_nb_max_value_properties += 3
        if max_nb_min_value_properties > 0:
            max_nb_min_value_properties += 3

        max_nb_extreme_properties = max(max_nb_min_value_properties,
                                        max_nb_max_value_properties)
        max_nb_extreme_properties = max(2, max_nb_extreme_properties)

        property_extreme_values = np.full((2, nb_properties,
                                       max_nb_extreme_properties),
                                      math.nan)

        for property_nb, property in enumerate(self.properties):
            min_value = property._min_value
            if property.closed_min:
                property_extreme_values[0, property_nb, 1] = 1
            else:
                property_extreme_values[0, property_nb, 1] = 0
            if type(min_value) in [float, int]:
                property_extreme_values[0, property_nb, 0] = min_value
            elif min_value is not None:
                if min_value.properties[0].closed_min:
                    property_extreme_values[0, property_nb, 0] = \
                        min_value.properties[0].min_value
                else:
                    property_extreme_values[0, property_nb, 0] = math.nan
                operation = min_value._operation
                if operation == "same_dimension_forward":
                    property_extreme_values[0, property_nb, 2] = 1
                for (min_val_nb,
                     min_val_property) in enumerate(min_value.properties):
                    property_extreme_values[0, property_nb,
                                            min_val_nb + 3] = min_val_property.number

            max_value = property._max_value
            # if closed, then use value 1, indicating that max should be
            # enforced, otherwise it just defines the geometry, which is
            # only available for position properties
            if property.closed_max:
                property_extreme_values[1, property_nb, 1] = 1
            else:
                property_extreme_values[1, property_nb, 1] = 0
            if type(max_value) in [float, int]:
                property_extreme_values[1, property_nb, 0] = max_value
            elif max_value is not None:
                if max_value.properties[0].closed_max:
                    property_extreme_values[1, property_nb, 0] = \
                        max_value.properties[0].max_value
                else:
                    property_extreme_values[1, property_nb, 0] = math.nan
                operation = max_value._operation
                if operation == "same_dimension_forward":
                    property_extreme_values[1, property_nb, 2] = 1
                for (max_val_nb,
                     max_val_property) in enumerate(max_value.properties):
                    property_extreme_values[1, property_nb,
                                        max_val_nb + 3] = max_val_property.number

        return property_extreme_values

    def _get_parameter_value_array(self, param_shape_batch, nb_timepoints,
                                   param_slice):
        nb_parameters = len(self.parameters)
        # parameter_value_shape = self.parameters[0].value_array.shape
        parameter_value_array = np.zeros((nb_parameters,
                                          nb_timepoints,
                                          param_shape_batch[1]))
        for nb, parameter in enumerate(self.parameters):
            array = torch.Tensor(parameter.value_array[:,param_slice])
            # array = array.unsqueeze(1)
            array = array.expand((1, nb_timepoints, param_shape_batch[1]))
            array = np.array(array.reshape((nb_timepoints, -1)))
            parameter_value_array[nb] = array
        return parameter_value_array

    def _get_transition_parameters(self):
        # - all transition rates
        nb_transitions = len(self.transitions)
        transition_parameters = np.full((nb_transitions, 2), np.nan)
        for nb, transition in enumerate(self.transitions):
            transition_parameters[nb, 0] = transition.parameter.number
            if transition.resources is not None:
                transition_parameters[nb, 1] = transition.resources
        return transition_parameters

    def _get_transition_state_arrays(self):
        nb_transitions = len(self.transitions)
        # - all transition start and end states
        all_transition_states = np.zeros((nb_transitions, 2))
        for nb, transition in enumerate(self.transitions):
            if transition.start_state is None:
                start_state = 0
            else:
                start_state = transition.start_state.number
            if transition.end_state is None:
                end_state = 0
            else:
                end_state = transition.end_state.number
            all_transition_states[nb, :] = (start_state, end_state)
        return all_transition_states

    def _get_transition_transferred_vals_array(self):
        nb_transitions = len(self.transitions)
        all_transition_tranferred_vals = np.full((nb_transitions, 2),
                                                 math.nan)
        for nb, transition in enumerate(self.transitions):
            transfer_property = transition.transfer_property
            if transfer_property is None:
                continue
            all_transition_tranferred_vals[nb, 0] = transfer_property[0].number
            all_transition_tranferred_vals[nb, 1] = transfer_property[1].number

        return all_transition_tranferred_vals

    def get_changed_start_values_for_creation(self):
        # get changed start values for transitions
        max_nb_changed_start_values = 0
        for transition_nb, transition in enumerate(self.transitions):
            if not hasattr(transition, "changed_start_values"):
                continue
            if transition.changed_start_values is None:
                continue
            nb_changed_start_values = len(transition.changed_start_values)
            max_nb_changed_start_values = max(max_nb_changed_start_values,
                                              nb_changed_start_values)

        # changed start values array has one row for each transition
        # and as many columns as max_nb_changed_start_values * 3
        # since each changed start value needs the property nb and up to
        # two columns for the new start value
        max_nb_changed_start_values = max(1,max_nb_changed_start_values)
        changed_start_values_array_shape = (len(self.transitions),
                                            len(self.properties),
                                            max_nb_changed_start_values*2)
        changed_start_values_array = np.full(changed_start_values_array_shape,
                                             math.nan)

        for transition_nb, transition in enumerate(self.transitions):
            if not hasattr(transition, "changed_start_values"):
                continue
            if transition.changed_start_values is None:
                continue
            if type(transition.changed_start_values) not in (list, tuple):
                changed_start_values = [transition.changed_start_values]
            else:
                changed_start_values = transition.changed_start_values

            for (start_val_nb,
                 changed_start_value) in enumerate(changed_start_values):
                property_nb = changed_start_value.object_property.number

                new_start_values = changed_start_value.new_start_values
                if type(new_start_values) in (tuple, list):
                    first_start_val = new_start_values[0]
                    changed_start_values_array[transition_nb,
                                               property_nb,
                                               0] = first_start_val
                    second_start_val = new_start_values[1]
                    changed_start_values_array[transition_nb,
                                               property_nb,
                                               1] = second_start_val
                else:
                    changed_start_values_array[transition_nb,
                                               property_nb,
                                               0] = new_start_values
        return changed_start_values_array

    def get_creation_on_objects_array(self):

        creation_on_objects = np.full(len(self.transitions), math.nan)
        for transition_nb, transition in enumerate(self.transitions):
            if not hasattr(transition, "creation_on_objects"):
                continue
            if transition.creation_on_objects:
                creation_on_objects[transition_nb] = 1
        return creation_on_objects

    def _get_transition_set_to_zero_properties(self):
        nb_transitions = len(self.transitions)
        max_nb_properties_set_to_zero = 0
        for nb, transition in enumerate(self.transitions):
            properties_set_to_zero = transition.properties_set_to_zero
            if properties_set_to_zero is None:
                continue
            max_nb_properties_set_to_zero = max(max_nb_properties_set_to_zero,
                                                len(properties_set_to_zero))
        max_nb_properties_set_to_zero = max(max_nb_properties_set_to_zero, 1)
        all_transition_set_to_zero_properties = np.full((nb_transitions,
                                                         max_nb_properties_set_to_zero),
                                                        math.nan)
        for nb, transition in enumerate(self.transitions):
            properties_set_to_zero = transition.properties_set_to_zero
            if properties_set_to_zero is None:
                continue
            property_nbs_set_to_zero = [prop.number
                                        for prop in properties_set_to_zero]
            all_transition_set_to_zero_properties[nb,
            :] = property_nbs_set_to_zero

        return all_transition_set_to_zero_properties

    def _get_action_parameter_array(self):
        # - all action rates
        nb_actions = len(self.actions)
        action_parameters = np.zeros((nb_actions))
        for action_nb, action in enumerate(self.actions):
            action_parameters[action_nb] = action.parameter.number
        return action_parameters

    def _get_action_properties_array(self):
        nb_actions = len(self.actions)
        all_action_properties = np.zeros((nb_actions, 1))
        for action_nb, action in enumerate(self.actions):
            all_action_properties[action_nb] = action.object_property.number
        return all_action_properties

    def _get_action_states_array(self):
        # - all states for each action
        nb_actions = len(self.actions)
        max_nb_action_states = 0
        nb_action_states = []
        for action in self.actions:
            if action.states is None:
                nb_states = 1
            else:
                nb_states = len(action.states)
            nb_action_states.append(nb_states)
            max_nb_action_states = max(max_nb_action_states,nb_states)

        action_state_array = np.zeros((nb_actions, max_nb_action_states))
        for action_nb, action in enumerate(self.actions):
            # for each action get list of length max_nb_action_states
            # fill the first elements with the actual states that
            # the action should be applied to and the remaining states
            # with nans (as no additional state)
            # 0 state means that action should be applied to all states
            nb_nans = max_nb_action_states - nb_action_states[action_nb]
            if action.states is None:
                states = [0]
            else:
                states = [state.number for state in action.states]
            states = [*states, *([math.nan] * nb_nans) ]
            action_state_array[action_nb] = states

        return action_state_array

    def _get_action_operation_array(self):
        nb_actions = len(self.actions)
        action_operation_array = np.zeros((nb_actions))
        for action_nb, action in enumerate(self.actions):
            if action._operation == "subtract":
                action_operation_array[action_nb] = -1
            else:
                action_operation_array[action_nb] = 1
        return action_operation_array

    def get_object_removal_property_array(self):
        # for object removal, there are two array with
        # several values for each condition
        # in all_object_removal_properties
        # the first value is the operation used to combine property values
        # with 1 meaning summing and -1 meaning subtracting
        # for subtracting all properties after the first one are subtracted
        # from the first one
        # the following values are property numbers
        # also give object removal function as parameter.
        max_nb_properties = 0
        for object_removal in self.object_removal:
            nb_properties_for_removal = len(object_removal.object_properties)
            max_nb_properties = max(max_nb_properties,
                                    nb_properties_for_removal)

        all_object_removal_properties = np.full((len(self.object_removal),
                                                 max_nb_properties+1),
                                                math.nan)

        for removal_nb, object_removal in enumerate(self.object_removal):
            object_removal_properties = [property.number
                                            for property
                                            in object_removal.object_properties]
            if object_removal.combine_properties_operation == "sum":
                all_object_removal_properties[removal_nb,:] = [1,
                                                               *object_removal_properties]
            elif object_removal.combine_properties_operation == "subtract":
                all_object_removal_properties[removal_nb,:] = [-1,
                                                               *object_removal_properties]
        return all_object_removal_properties

    def _get_object_removal_property_array(self):
        # for the second array, object_removal_operations
        # the first value is the operation, which can be 1 for the combined
        # property value being larger then a threshold to be removed
        # and -1 for the combined property value being smaller then a threshold
        # to be removed
        # the second value is the threshold
        object_removal_operations = np.zeros((len(self.object_removal),
                                              2))
        for removal_nb, object_removal in enumerate(self.object_removal):
            if object_removal.compare_to_threshold == "smaller":
                object_removal_operations[removal_nb, 0] = -1
            else:
                object_removal_operations[removal_nb, 0] = 1

            object_removal_operations[removal_nb, 1] = object_removal.threshold
        return object_removal_operations

    def get_first_last_idx_with_object(self, object_states):
        # get an array to save the highest index with an object
        idx_array = np.expand_dims(np.arange(self.object_states.shape[0]),
                                   axis=[nb+1 for nb
                                         in
                                         range(len(self.object_states.shape[1:])
                                               )])
        object_state_mask = object_states[0] > 0
        highest_idx_with_object = (object_state_mask * idx_array).max(axis=0)
        highest_idx_with_object = np.expand_dims(highest_idx_with_object,0)

        object_state_mask = object_states[0] == 0
        lowest_idx_no_object = (object_state_mask * idx_array).min(axis=0)
        lowest_idx_no_object = np.expand_dims(lowest_idx_no_object,0)

        first_last_idx_with_object = np.concatenate([lowest_idx_no_object,
                                                     highest_idx_with_object],
                                                    axis=0)
        return first_last_idx_with_object

    def _get_param_prop_dependence_array(self):
        """
        # implement linearly changing parameters,
        # as added to normal/baseline parameter value
        # as two values: value at start and/or end of neurite,
        # if both are defined, then the change per um is calculated
        # and definitions of change per um etc are not taken into
        # account
        # if one of the values is defined then the change from that
        # position (start or end of neurite) is defined by:
        # absolute or relative change per um or per % of length
        # Thereby, the added linearly changing part might even go to 0
        # array shape:
        # (nb_params,
        #  5 (2 (start+end) + 1 (abs/rel of parameter)
        #     + 1 (um/% of length) + 1 (change) + 1 (idx in object_rates),
        #  param_value_shape)
        # for different simulations all parameters can be defined
        # separately.

        # For position dependent rates, to choose the correct object
        # for a state transition, the position of the object has to
        # be taken into account - not every object has the same chance
        # of transitioning. Therefore, the position dependent rate of
        # an object must be saved.
        # A parameter might be used in more than one transition and
        # therefore the same position dependence might influence more
        # than one transition. However, since the position dependence
        # is exactly the same, the position dependent rate will also be
        # the same for that object. Only the baseline rate might be
        # different. Therefore, one object value per dependence is
        # sufficient.
        """

        max_nb_properties = 0
        for param_nb, parameter in enumerate(self.parameters):
            if parameter.dependence is None:
                continue
            dependence = parameter.dependence
            if dependence.properties is None:
                nb_properties = len(self.properties)
            else:
                nb_properties = len(dependence.properties)
            max_nb_properties = max(max_nb_properties, nb_properties)

        params_prop_dependence = np.full((len(self.parameters),
                                         7 + max_nb_properties),
                                        np.nan)

        position_dependence = False
        dependence_nb = 0
        for param_nb, parameter in enumerate(self.parameters):
            if parameter.dependence is None:
                continue
            position_dependence = True
            # so far only linear dependence on space are allowed
            dependence = parameter.dependence
            if type(dependence.start_val) == type(self.parameters[0]):
                start_val_param_nb = dependence.start_val.number
                params_prop_dependence[param_nb, 0] = start_val_param_nb
            if type(dependence.end_val) == type(self.parameters[0]):
                end_val_param_nb = dependence.end_val.number
                params_prop_dependence[param_nb, 1] = end_val_param_nb
            # if the start and end values are defined but no change
            # calculate the linear change from the start to the end val
            if ((dependence.start_val is not None) &
                    (dependence.end_val is not None) &
                    (dependence.param_change is None)):
                # use absolute changes of parameter
                params_prop_dependence[param_nb, 2] = 0
                # change is per um length
                params_prop_dependence[param_nb, 3] = 0

                # start_val_param = dependence.start_val
                # end_val_param = dependence.end_val.number
                # param_diff = (start_val_param.value_array -
                #               end_val_param.value_array)
                # abs_change = param_diff / self.properties[0].max_value
                params_prop_dependence[param_nb, 4] = math.nan
                dependence_nb += 1
                continue

            # if rel_change is 0, then the change is absolute
            if dependence.param_change_is_abs:
                rel_param_change = 0
            else:
                rel_param_change = 1
            params_prop_dependence[param_nb, 2] = rel_param_change

            if dependence.prop_change_is_abs:
                rel_length_change = 0
            else:
                rel_length_change = 1
            params_prop_dependence[param_nb, 3] = rel_length_change

            change_param = dependence.param_change
            params_prop_dependence[param_nb, 4] = change_param.number

            params_prop_dependence[param_nb, 5] = dependence_nb

            if dependence.function == "exponential":
                function = 1
            else:
                function = 0

            params_prop_dependence[param_nb, 6] = function

            # if properties are defined, add the property numbers of all
            # defined properties
            if dependence.properties is not None:
                for property_nb, property in enumerate(dependence.properties):
                    params_prop_dependence[param_nb,
                                         7 + property_nb] = property.number
            else:
                # if properties are not defined, add all property numbers to
                # the list
                params_prop_dependence[param_nb, 7:] = list(range(len(self.properties)))

            dependence_nb += 1

        return params_prop_dependence, position_dependence, dependence_nb

    def _start(self, nb_simulations, min_time, data_extractions,data_folder,
               time_resolution, save_initial_state=False,
               max_number_objects=None, all_parameter_combinations=False,
               single_parameter_changes=True, nb_parallel_cores=1, seed=42,
               ignore_errors=False,
               print_update_time_step=1, nb_objects_added_per_step=10,
                max_iters_no_data_extraction=2000,
               dynamically_increase_nb_objects=False,
               remove_finished_sims=False, save_states=False,
               save_results=True, print_times=True,
               use_assertion_checks=True, reset_folder=True,
               bug_fixing=False):
        """

        Args:
            data_extractions (Dict of DataExtraction objects): Dict in which
                values are different DataExtraciton objects. Keys are names
                of extracted data.
            nb_simulations (int): Number of simulations to run per parameter
                combination
            min_time (float): minimum time
            max_number_objects (int): maximum number of objects allowed to be
                simulated. Determines array size
            all_parameter_combinations (Boolean): Whether the combination of all
                supplied parameter values from all parameters should be
                analyzed. This means that every value of every parameter is
                combined with every value of every other parameter and thereby
                simulations of all parameter combinations are executed.
            single_parameter_changes (Boolean): Whether parameter values should
                be used as changing one parameter value at a time from the
                standard value (first value). This way looking at changes of
                single parameters can be executed easily.
                If False, then each simulation will use parameter values from
                one index for all supplied parameters. Therefore, lists
                of values for all parameters have to have the same length -
                or the can have only one value, in which case this single value
                will be expanded to the total length of parameter values.
                This allows e.g. definition of pairs of parameter values to use
                (changing two parameters for a simulation) but also any other
                combination of parameter value changes.
        Returns:
        """
        if reset_folder:
            if os.path.exists(data_folder):
                # delete all files except scripts
                # thereby prevent the executing script from being deleted
                # if the script inside the folder is executed
                for file_to_remove in os.listdir(data_folder):
                    if file_to_remove.find(".py") == -1:
                        path_to_remove = os.path.join(data_folder,
                                                      file_to_remove)
                        if os.path.isdir(path_to_remove):
                            shutil.rmtree(path_to_remove)
                        else:
                            os.remove(path_to_remove)

                time.sleep(0.1)
            # os.mkdir(data_folder)

        if not os.path.exists(data_folder):
            os.mkdir(data_folder)

        self.min_time = min_time
        self.data_extractions = data_extractions
        self.data_folder = data_folder
        self.ignore_errors = ignore_errors
        self.nb_objects_added_per_step = nb_objects_added_per_step
        self.dynamically_increase_nb_objects = dynamically_increase_nb_objects
        self.save_states = save_states
        self.save_results = save_results
        self.print_times = print_times
        self.max_number_objects = max_number_objects
        self.use_assertion_checks = use_assertion_checks
        self.max_iters_no_data_extraction = max_iters_no_data_extraction
        self.remove_finished_sims = remove_finished_sims
        self.all_parameter_combinations = all_parameter_combinations
        self.single_parameter_changes = single_parameter_changes
        self.time_resolution = time_resolution

        if (not single_parameter_changes) & self.all_parameter_combinations:
            print("If the parameters single_parameter_changes is False, "
                  "no parameters will be combined, also if "
                  "all_parameter_combinations is True.")


        self.all_data = []
        self.all_times = []
        self.all_states = []
        self.all_iteration_nbs = []

        self.times_buffer = []
        self.object_states_buffer = []

        self.initial_memory_used = 0
        self.iteration_memory_used = 0

        self.all_removed_positions = pd.DataFrame()

        self._get_all_parameters()

        # create list with all transitions and then with all actions
        self._all_simulation_parameters = [*self.states, *self.parameters]

        simulation_parameter_lengths = self._get_simulation_parameter_lengths()

        # array size contains for each combination of parameters to explore
        self._simulation_array_size = [self.max_number_objects, nb_simulations,
                                       *simulation_parameter_lengths]

        # get number of timepoints
        nb_timepoints = math.ceil(self.min_time / time_resolution)
        if save_initial_state:
            nb_timepoints += 1

        self._initialize_parameter_arrays(self.parameters,
                                          simulation_parameter_lengths,
                                          single_parameter_changes,
                                          all_parameter_combinations,
                                          nb_timepoints)

        # create index array in which each entry has the value of the index of
        # the microtubule in the simulation, thereby multi-D operations
        # on single simulations can be executed
        # create array of index numbers of same shape as whole simulation array
        view_array = [-1] + [1] * (len(self._simulation_array_size) - 1)
        max_number_objects = self.max_number_objects

        self.index_array = torch.linspace(1, max_number_objects,
                                          max_number_objects,
                                          dtype=torch.int16).view(*view_array)

        # self.get_tensor_memory()

        if dynamically_increase_nb_objects:
            self._add_objects_to_full_tensor()

        # data = self.data_extraction.extract(self)
        #
        # if self.device.find("cuda") != -1:
        #     self._free_gpu_memory()
        #
        # self._save_data(data, 0)


        # self._add_data_to_buffer()
        self._save_simulation_parameters()

        self.data_buffer = []
        self.last_data_extraction = 0
        self.last_data_saving = 0

        # to make run_iteration compatible with numba.njit(debug=True) (numba no python)
        # use numpy arrays for everything instead of python objects
        # create one array for

        # create arrays with the parameter size to be explored in one round


        property_start_values = self._get_property_start_val_arrays()

        properties_array = self._get_property_vals_array()

        property_extreme_values = self._get_property_extreme_val_arrays()

        transition_parameters = self._get_transition_parameters()
        all_transition_states = self._get_transition_state_arrays()
        get_set_zero_array = self._get_transition_set_to_zero_properties()
        all_transition_set_to_zero_properties = get_set_zero_array
        get_transfer_vals_array = self._get_transition_transferred_vals_array()
        all_transition_tranferred_vals = get_transfer_vals_array

        get_changed_start_values = self.get_changed_start_values_for_creation
        changed_start_values_array = get_changed_start_values()
        creation_on_objects = self.get_creation_on_objects_array()

        action_parameters = self._get_action_parameter_array()
        all_action_properties = self._get_action_properties_array()
        action_operation_array = self._get_action_operation_array()
        action_state_array = self._get_action_states_array()

        all_object_removal_properties = self.get_object_removal_property_array()
        object_removal_operations = self._get_object_removal_property_array()

        parameter_shape = simulation_parameter_lengths
        # nb_simulations = self.object_states.shape[1]

        start_nb_parallel_cores = nb_parallel_cores

        # current_min_time = np.min(self.times)
        # if current_min_time >= min_time:
        #     break
        convert_array = lambda x: np.ascontiguousarray(x)

        time_resolution = convert_array(time_resolution)

        device = "gpu"

        nb_parameter_combinations = np.sum(simulation_parameter_lengths)

        local_resolution = 0.02
        local_density_size = int(20 / local_resolution) + 1

        if np.nansum(creation_on_objects) > 0:
            some_creation_on_objects = True
        else:
            some_creation_on_objects = False

        nb_parallel_cores = np.zeros((nb_simulations,
                                      *simulation_parameter_lengths))
        nb_parallel_cores[:,:] = start_nb_parallel_cores

        local_density_batches = None
        total_density_batches = None

        (params_prop_dependence,
         position_dependence,
         nb_dependences) = self._get_param_prop_dependence_array()

        print(0, numba.cuda.current_context().get_memory_info()[
            0] / 1024 / 1024 / 1024)
        if self.device.find("cuda") != -1:

            start = time.time()

            to_cuda = cuda.to_device

            # get number of cuda stream managers and cores per stream manager
            nb_SM, nb_cc = self._get_number_of_cuda_cores()

            (rng_states,
             simulation_factor,
             parameter_factor) = self._get_random_number_func(nb_parameter_combinations,
                                                              nb_simulations,
                                                              seed)

            # get free GPU memory if device is cuda, to do calculations in batches
            # that fit on the GPU
            (free_memory,
             total_memory) = numba.cuda.current_context().get_memory_info()

            # get size of all arrays that were created
            # the size depends on

            # get size of object states

            object_state = torch.zeros((1,), dtype=torch.int8)
            size_states = (np.prod(self._simulation_array_size,
                                   dtype=np.int64) *
                           object_state.element_size())

            # get size of properties times the number of properties+
            object_property = torch.zeros((1,), dtype=torch.float32)
            size_properties = (np.prod((len(self.properties),
                                        *self._simulation_array_size),
                                       dtype=np.int64) *
                               object_property.element_size())

            # object_states = np.array(self.object_states)
            size_density = (np.product((3, local_density_size, nb_simulations,
                                        nb_parameter_combinations),
                                               dtype=np.int64) *
                            object_property.element_size())

            size_tminmax = (np.product((2, len(self.properties),
                                        *self._simulation_array_size),
                                       dtype=np.int64) *
                            object_property.element_size())

            size_property_changes = (np.product((2, len(self.properties),
                                                *self._simulation_array_size),
                                               dtype=np.int64) *
                                     object_property.element_size())

            size_object_dep_rates = (np.product((nb_dependences,
                                               *self._simulation_array_size
                                               ), dtype=np.int64)
                                     * object_property.element_size())


            total_size = ((nb_timepoints+1) * (size_states + size_properties)
                          + size_density + size_tminmax
                          + size_property_changes + size_object_dep_rates)*1.4

            # think about sorting parameter values in the array by
            # their magnitude - higher transition rates should be separate from
            # lower transition rates to allow for splitting simulation batches
            # into higher and lower rates. Thereby, all simulations with lower
            # rates will be faster, while only the simulations with higher
            # rates will be slower.
            # If each batch would contain higher and lower rates, all batches
            # would take longer.

            # if total size is larger than free memory on GPU
            # split array by number of parameters into smaller pieces
            nb_memory_batches = int(math.ceil(total_size/free_memory))

            # also check how many parameter combinations can be processed
            # with the number of cuda cores available
            # total_nb_simulations = (self.object_states.shape[1] *
            #                         self.object_states.shape[2])
            # nb_core_batches = int(math.ceil(self.total_nb_simulations / (nb_SM *
            #                                                         nb_cc)))

            nb_batches = max(nb_memory_batches, 1)

            param_combinations_per_batch = max(1,
                                               int(math.floor(
                                                   (nb_parameter_combinations
                                                    / nb_batches))))

            # if there is more than one batch and the number of parameter
            # combinations is not divisably by the number of batches,
            # add one to the number of batches
            if (nb_batches > 1) & (nb_parameter_combinations % nb_batches != 0):
                nb_batches += 1

            simulation_numba._decorate_all_functions_for_gpu(self)
            object_states_batches = None
            properties_batches_array = None
            times_batches = None

            if nb_batches > nb_parameter_combinations:
                raise ValueError("GPU memory too small to support all "
                                 "simulations of one parameter value. Batching "
                                 "is only implemented for parameter values. "
                                 "reduce the number of simulations or use a "
                                 "GPU with more memory.")

            # nb_batches = 2
            # param_combinations_per_batch = 500

            nb_parallel_cores = to_cuda(convert_array(nb_parallel_cores))

            # are property arrays expanded upon definition?
            # are parameter value arrays expanded upon definition?

            param_shape_batch = (nb_simulations,
                                 param_combinations_per_batch)

            print(f"Simulating {nb_batches} batches... \n")
            # print(1, numba.cuda.current_context().get_memory_info()[0]/1024/1024/1024)
            for batch_nb in tqdm.tqdm(range(nb_batches)):

                start_parameter_comb = (batch_nb *
                                        param_combinations_per_batch)
                end_parameter_comb = (start_parameter_comb +
                                      param_combinations_per_batch)
                param_slice = slice(start_parameter_comb, end_parameter_comb)

                parameter_value_array = self._get_parameter_value_array(param_shape_batch,
                                                                        nb_timepoints,
                                                                        param_slice)

                # for numberr in range(parameter_value_array.shape[0]):
                #     print(parameter_value_array[numberr])

                object_dependent_rates = np.zeros((nb_dependences,
                                                   max_number_objects,
                                                   *param_shape_batch
                                                   ))

                self._initialize_object_states(param_shape_batch)

                self._initialize_object_properties(param_shape_batch)

                self.object_states = self.object_states.reshape(
                    (self.object_states.shape[0],
                     self.object_states.shape[1],
                     -1))

                for property in self.properties:
                    property.array = property.array.reshape(
                        (property.array.shape[0],
                         property.array.shape[1],
                         -1))
                    property.array = np.array(property.array)

                self.times = torch.zeros((1, *param_shape_batch))

                # reshape to have all different parameter values in one axis
                # with the first axis being the number of objects
                self.times = np.array(self.times.reshape((self.times.shape[0],
                                                          self.times.shape[1],
                                                          -1)))

                # all parameters for nucleation on objects
                nb_properties = len(self.properties)
                end_position_tmax_array = np.zeros((32,
                                                    nb_properties - 1,
                                                    *param_shape_batch))
                properties_tmax_array = np.zeros((32,
                                                  nb_properties - 1,
                                                  nb_properties,
                                                  *param_shape_batch))
                # first dimensions is unsorted and sorted
                properties_tmax = np.full((2, 32,
                                           nb_properties - 1, *param_shape_batch),
                                          math.nan)
                # properties_tmax_sorted = np.full((32,
                #                                   nb_properties-1, *param_shape_batch),
                #                                  math.nan)
                current_sum_tmax = np.zeros((32,
                                             nb_properties, *param_shape_batch))
                property_changes_tminmax_array = np.zeros(
                    (2, max_number_objects,
                     nb_properties,
                     *param_shape_batch),
                    dtype=np.float32)

                property_changes_per_state = np.zeros(
                    (len(self.states), nb_properties,
                     *param_shape_batch))
                nucleation_changes_per_state = np.copy(
                    property_changes_per_state)
                total_property_changes = np.zeros(
                    (len(self.states), *param_shape_batch))

                density_threshold_boundaries = np.zeros((3, *param_shape_batch))

                # -  current rates of all transitions (empty)
                nb_transitions = len(self.transitions)
                current_transition_rates = np.zeros((nb_transitions,
                                                     *param_shape_batch))
                # print(action_parameters, transition_parameters,
                #       current_parameter_values.shape,
                #       parameter_value_array.shape)

                # - all total transition rates (empty)
                total_rates = np.zeros(param_shape_batch)

                # - current reaction times (empty)
                reaction_times = np.zeros(param_shape_batch)

                current_transitions = np.zeros(param_shape_batch)

                all_transition_positions = np.zeros(param_shape_batch)

                # create new object states object that includes data for all timepoints
                object_states = np.zeros((nb_timepoints + 2,
                                          self.object_states.shape[0],
                                          *param_shape_batch),
                                         dtype=np.int32)

                object_states[0] = self.object_states

                if save_initial_state:
                    object_states[1] = self.object_states

                # calculate number of objects for all states
                nb_objects_all_states = np.zeros((2, max(len(self.transitions),
                                                      len(self.states)) + 1,
                                                  *param_shape_batch))
                nb_objects_all_states[0, 0] = self.object_states.shape[0]

                for state_nb, state in enumerate(self.states):
                    nb_objects_all_states[0, state_nb + 1] = np.sum(
                        object_states[0] == state.number, axis=0)

                nb_properties = len(self.properties)

                properties_array = np.full((nb_timepoints + 1, nb_properties,
                                            self.object_states.shape[0],
                                            *param_shape_batch),
                                           math.nan)

                if save_initial_state:
                    for property_nb, property in enumerate(self.properties):
                        properties_array[1, property_nb] = property.array


                local_density = np.zeros((3, local_density_size, nb_simulations,
                                          param_combinations_per_batch))

                total_density = np.zeros(
                    (nb_simulations, param_combinations_per_batch))

                # first idx of first dimension is for constant term,
                # second idx of first dimension is for second order term
                # first idx of second dimension is for base part, second idx is for
                # variable part
                tau_square_eq_terms = np.zeros(
                    (2, 2, nb_simulations, param_combinations_per_batch))

                thread_masks = np.zeros((4, *param_shape_batch), dtype=np.int64)

                # set arrays for values over time
                # the first index is the current timepoint with respect to
                # time_resolution
                # the second index with respect to the timestep for reassigning threads
                # from the third idx the actual timepoints at the respective savepoint
                # are saved
                timepoint_array = np.full((nb_timepoints + 2,
                                           *param_shape_batch),
                                          math.nan)

                timepoint_array[:1] = 0

                if save_initial_state:
                    timepoint_array[2] = 0

                timepoint_array = convert_array(timepoint_array)
                properties_array = convert_array(properties_array)
                first_last_idx_with_object = self.get_first_last_idx_with_object(
                    object_states)

                size = min(nb_simulations * param_combinations_per_batch
                           * start_nb_parallel_cores,
                           int((nb_SM * nb_cc)))

                thread_to_sim_id = np.zeros((size, 3))

                """
                Batched data:

                Initial state dependent:
                object_states
                properties_array
                nb_objects_all_states
                all_transition_positions
                # for batching don't allow initial states
                
                Should additionally be batched:
                
                initial cond dependent:
                first_last_idx_with_object
                """

                local_density_batch = cuda.to_device(
                    convert_array(local_density))
                total_density_batch = cuda.to_device(
                    convert_array(total_density))

                timepoint_array_batch = cuda.to_device(
                    convert_array(timepoint_array))
                time_resolution = cuda.to_device(time_resolution)

                sim = simulation_numba._execute_simulation_gpu

                object_states_batch = convert_array(object_states)
                object_states_batch = to_cuda(object_states_batch)

                object_dependent_rates_batch = object_dependent_rates
                object_dependent_rates_batch = convert_array(to_cuda(
                    object_dependent_rates_batch))

                property_array_batch = properties_array
                property_array_batch = convert_array(property_array_batch)
                property_array_batch = to_cuda(property_array_batch)


                times_batch = self.times
                param_val_array_batch = parameter_value_array
                current_trans_rates = current_transition_rates
                nb_obj_all_states_batch = nb_objects_all_states

                total_rates_batch = total_rates
                reaction_times_batch = reaction_times
                current_transitions_batch = current_transitions
                all_trans_pos_batch = all_transition_positions

                first_last_idx_with_object = to_cuda(convert_array(first_last_idx_with_object))
                numba.cuda.profile_start()

                # print(params_prop_dependence)
                # print(param_val_array_batch[:,0,0])

                # print("Starting simulation batch...")
                sim[nb_SM,
                 nb_cc](object_states_batch, #int32[:,:,:]
                            property_array_batch, #float32[:,:,:,:]
                            to_cuda(convert_array(times_batch)), #float32[:,:,:]
                            nb_simulations, #int32
                            param_combinations_per_batch, #int32
                            to_cuda(convert_array(param_val_array_batch)), #float32[:,:,:]
                            to_cuda(convert_array(params_prop_dependence)),
                            position_dependence,
                            object_dependent_rates_batch,
                            to_cuda(convert_array(transition_parameters)), #int32[:]
                            to_cuda(convert_array(all_transition_states)), #int32[:,:]
                            to_cuda(convert_array(action_parameters)), #int32[:]
                            to_cuda(convert_array(action_state_array)), #float32[:,:]
                            to_cuda(convert_array(all_action_properties)), #float32[:,:]
                            to_cuda(convert_array(action_operation_array)), #float32[:,:]
                            to_cuda(convert_array(current_trans_rates)), #float32[:,:,:,:]
                            to_cuda(convert_array(property_start_values)), #float32[:,:]
                            to_cuda(convert_array(property_extreme_values)), #float32[:,:,:]
                            to_cuda(convert_array(
                                all_transition_tranferred_vals)), #float32[:,:]
                            to_cuda(convert_array(
                                all_transition_set_to_zero_properties)), #float32[
                            to_cuda(convert_array(changed_start_values_array)),
                            to_cuda(convert_array(creation_on_objects)),
                            some_creation_on_objects,
                            to_cuda(convert_array(
                                all_object_removal_properties)),
                            to_cuda(convert_array(object_removal_operations)),

                            to_cuda(convert_array(nb_obj_all_states_batch)),

                            # all arrays with one property per simulation
                            # can be potentially combined, but at the cost
                            # of less readable code
                            to_cuda(convert_array(total_rates_batch)),
                            to_cuda(convert_array(reaction_times_batch)),
                            to_cuda(convert_array(current_transitions_batch)),
                            to_cuda(convert_array(all_trans_pos_batch)),

                            # all parameters for nucleation on objects
                            to_cuda(convert_array(end_position_tmax_array)),
                            to_cuda(convert_array(properties_tmax_array)),
                            to_cuda(convert_array(properties_tmax)),
                            to_cuda(convert_array(current_sum_tmax)),

                            to_cuda(convert_array(property_changes_tminmax_array)),
                            to_cuda(convert_array(property_changes_per_state)),
                            to_cuda(convert_array(nucleation_changes_per_state)),
                            to_cuda(convert_array(total_property_changes)),
                            to_cuda(convert_array(density_threshold_boundaries)),

                            to_cuda(convert_array(tau_square_eq_terms)),
                            first_last_idx_with_object,

                           timepoint_array_batch,
                           time_resolution, min_time, save_initial_state,

                            start_nb_parallel_cores,
                            nb_parallel_cores,
                            to_cuda(convert_array(thread_masks)),
                            to_cuda(convert_array(thread_to_sim_id)),

                            local_density_batch, local_resolution,
                        total_density_batch,

                           rng_states, simulation_factor, parameter_factor
                             )
                if self.print_times:
                    print("Simulation time: ", np.round(time.time() - start, 2), "\n")

                # nb_parallel_cores = nb_parallel_cores.copy_to_host()


                numba.cuda.synchronize()
                numba.cuda.profile_stop()

                object_state_batch = torch.Tensor(
                    object_states_batch.copy_to_host()[2:])
                property_array_batch = torch.Tensor(
                    property_array_batch.copy_to_host()[1:])
                times_batch = torch.Tensor(timepoint_array_batch.copy_to_host()[2:])

                first_last_idx_with_object = first_last_idx_with_object.copy_to_host()

                del timepoint_array_batch
                cuda.current_context().memory_manager.deallocations.clear()

                # plt.figure()
                # plt.plot(local_density_batches.mean(axis=1))

                self.object_states = object_state_batch

                self.times = (times_batch * time_resolution.copy_to_host())
                self.times = self.times.unsqueeze(1).to(self.device)

                for property_nb, property in enumerate(self.properties):
                    property.array = property_array_batch[:, property_nb]
                    property.array = property.array.to(self.device)

                # self.object_states = object_states_batch#.to(self.device)

                # print(111, self.object_states.shape)
                # print(222, property_array_batch.shape)

                # nb_objects_single = torch.count_nonzero(self.object_states,
                #                                         dim=1).to(torch.float)
                # nb_objects = nb_objects_single.mean(dim=-2)
                # for parameter_nb in range(nb_objects.shape[-1]):
                #     print("Parameters: ")
                #     for parameter in self.parameters:
                        # standard_param_val = parameter.value_array[0, 0].item()
                        # param_val = parameter.value_array[
                        #     0, parameter_nb].item()
                        # param_val_string = ""
                        # if standard_param_val != param_val:
                        #     param_val_string += "!! "
                        # param_val_string += parameter.name
                        # param_val_string += ": "
                        # param_val_string += str(param_val)
                        # param_val_string += "; "
                        # print(param_val_string)
                    # print("\n")

                all_data = {}
                for data_name, data_extraction in self.data_extractions.items():
                    start = time.time()
                    all_data[data_name] = data_extraction.extract(self,
                                                                  first_last_idx_with_object=
                                                                  first_last_idx_with_object)
                    if self.print_times:
                        print(data_name, "extraction time: ",
                              np.round(time.time() - start, 2))

                    if data_extraction.show_data:
                        print_data = False
                        for sub_name, sub_data in all_data[
                            data_name].items():
                            if sub_data.sum() == 0:
                                continue
                            if sub_data.shape[1] == 1:
                                print_data = True
                                break
                            if sub_name.endswith("_position"):
                                continue
                            if sub_name.find("_bins_") != -1:
                                continue
                            all_axs = []
                            max_y = 0
                            for timepoint in range(sub_data.shape[0]):
                                # the first dimension is for the time
                                # only plot data for the last timepoint
                                plot_data = torch.mean(sub_data[timepoint].cpu(),
                                                       dim=(1))
                                plt.figure()
                                found_param_change = False
                                for param_nb in range(plot_data.shape[1]):
                                    plot_data_param = plot_data[:,param_nb].unsqueeze(1)
                                    param_val_string = ""
                                    for parameter in self.parameters:
                                        standard_param_val = parameter.value_array[0, 0].item()
                                        param_val = parameter.value_array[0, param_nb].item()
                                        if standard_param_val != param_val:
                                            param_val_string = (parameter.name +
                                                                ": " +
                                                                str(param_val))
                                            found_param_change = True
                                            break
                                    # if (len(plot_data.shape) == 1):
                                    #     plot_data = plot_data.unsqueeze(1)
                                    # if plot_data.shape[1] > 1:
                                    #     plot_data = plot_data[:, 0]
                                    torch.set_printoptions(precision=3,
                                                           sci_mode=False)
                                    # if len(plot_data) < 30:
                                    #     print(plot_data)
                                    #     print(plot_data[:-1] / plot_data[1:])
                                    #     print("\n")

                                    plt.plot(plot_data_param,
                                             label=param_val_string)

                                    max_y = max(max_y, plt.ylim()[1])
                                    all_axs.append(plt.gca())
                                    # plt.figure()
                                    # plt.plot(torch.mean(sub_data[-1],dim=(1)))
                                    # plt.ylim(0,3.5)
                                # plt.ylim(0, max_y)
                                if found_param_change:
                                    plt.legend()
                                plt.title(data_name + "-" + sub_name +
                                          "; Time: " + str(timepoint))
                                # to plot total values over time
                                # print("Time ", timepoint, " : ", plot_data_param.sum())
                            for ax in all_axs:
                                ax.set_ylim(0, max_y)
                        if not print_data:
                            continue
                        for sub_name, sub_data in all_data[
                            data_name].items():
                            if ((sub_name.find("mean") == -1) &
                                    (sub_name.find("number") == -1)):
                                continue
                            # if sub_name.find("inside") == -1:
                            #     continue
                            mean = np.nanmean(sub_data[-1].cpu())
                            if mean == 0:
                                continue
                            if np.isnan(mean):
                                continue
                            print(sub_name, mean)

                print("Saving raw and analyzed data ...")
                if self.save_results:
                    self._save_times_and_object_states(batch_nb)
                    self._save_data(all_data, batch_nb)

                # print(2, numba.cuda.current_context().get_memory_info()[0]/1024/1024/1024)
                if nb_batches > 1:
                    self.data_buffer = []
                    del self.times
                    del self.object_states
                    del self.object_states_buffer

                    for property in self.properties:
                        property.array = []
                    # del all_data

                    del object_dependent_rates_batch
                    del property_array_batch
                    del object_states_batch
                    del local_density_batch
                    del total_density_batch
                else:
                    self.all_data = all_data
                cuda.current_context().memory_manager.deallocations.clear()
                torch.cuda.empty_cache()
                print(3, numba.cuda.current_context().get_memory_info()[0]/1024/1024/1024)

            del rng_states
            cuda.current_context().memory_manager.deallocations.clear()

        else:
            if not bug_fixing:
                simulation_numba._decorate_all_functions_for_cpu()
            else:
                simulation_numba._reassign_random_number_func_cpu()

            print("Starting simulation...")
            start = time.time()

            _execute_sim = simulation_numba._execute_simulation_cpu
            _execute_sim(convert_array(object_states),
                        convert_array(properties_array),
                        convert_array(self.times),
                        nb_simulations, nb_parameter_combinations,
                        convert_array(parameter_value_array),
                        convert_array(transition_parameters),
                        convert_array(all_transition_states),
                        convert_array(action_parameters),
                        convert_array(action_state_array),
                        convert_array(all_action_properties),
                        convert_array(action_operation_array),
                        convert_array(current_transition_rates),
                        convert_array(property_start_values),
                        convert_array(property_extreme_values),
                        convert_array(all_transition_tranferred_vals),
                        convert_array(all_transition_set_to_zero_properties),
                        convert_array(changed_start_values_array),
                        convert_array(creation_on_objects),
                        some_creation_on_objects,
                        convert_array(all_object_removal_properties),
                        convert_array(object_removal_operations),

                        convert_array(nb_objects_all_states),
                        convert_array(total_rates),
                        convert_array(reaction_times),
                        convert_array(current_transitions),
                        convert_array(all_transition_positions),

                         # all parameters for nucleation on objects
                         convert_array(end_position_tmax_array),
                         convert_array(properties_tmax_array),
                         convert_array(properties_tmax),
                         convert_array(current_sum_tmax),
                         convert_array(property_changes_tminmax_array),
                         convert_array(property_changes_per_state),
                            convert_array(nucleation_changes_per_state),
                         convert_array(total_property_changes),
                         convert_array(density_threshold_boundaries),

                         convert_array(tau_square_eq_terms),

                         convert_array(first_last_idx_with_object),

                        timepoint_array,
                        time_resolution, min_time, save_initial_state,

                         convert_array(nb_parallel_cores),
                         convert_array(thread_masks),

                         local_density, local_resolution,
                         convert_array(total_density),
                         seed
                        )

            if self.print_times:
                print("Simulation time: ", time.time() - start)

            self.object_states = torch.Tensor(np.copy(object_states[1:]))

            for property_nb, property in enumerate(self.properties):
                property.array = torch.Tensor(np.copy(
                    properties_array[1:,:, property_nb]))

            self.times = (torch.Tensor(np.copy(timepoint_array)).unsqueeze(1)
                          * time_resolution)

        self._save_all_metadata(nb_simulations, max_number_objects)
        print("Finished saving all data.")
        # self.all_data = all_data
        # self.times = times

    def _get_all_parameters(self):
        # get different parameters from all events#, so that no parameter is
        # used twice (some transitions may have the same parameter)
        parameters = [event.parameter for event in [*self.transitions,
                                                    *self.actions]]
        param_nb = 0
        self.parameters = []
        for parameter in parameters:
            if parameter.number is None:
                parameter.number = param_nb
                if parameter.name == "":
                    parameter.name = str(parameter.number)
                param_nb += 1
                self.parameters.append(parameter)

        # first get all unique dependencies
        self.dependencies = []
        dependence_nb = 0
        for parameter in self.parameters:
            dependence = parameter.dependence
            if dependence is None:
                continue
            dependence.number = dependence_nb
            if dependence.name == "":
                dependence.name = str(dependence.number)
            dependence_nb += 1
            self.dependencies.append(dependence)

        # then go through each dependency and extract parameters
        for dependence in self.dependencies:
            all_potential_params = [dependence.start_val, dependence.end_val,
                                    dependence.param_change,
                                    dependence.param_change_is_abs,
                                    dependence.prop_change_is_abs]
            for potential_param in all_potential_params:
                if type(potential_param) == type(self.parameters[0]):
                    # check whether parameter was already used somewhere else
                    # and therefor is already in the self.parameters array
                    # by checking whether the number is defined
                    # (as done above when getting the first set of parameters)
                    if potential_param.number is None:
                        potential_param.number = param_nb
                        self.parameters.append(potential_param)
                        param_nb += 1

    def _get_simulation_parameter_lengths(self):
        # create list of length of all model parameters
        state_parameter_lengths = [len(parameters.values)
                                        for parameters
                                        in self.states]

        parameter_lengths = [len(parameters.values[0])
                                        for parameters
                                        in self.parameters]

        simulation_parameter_lengths = [*state_parameter_lengths,
                                        *parameter_lengths]

        # check whether each parameter either only has a single value
        # or has a length that is similar between all parameters with more than
        # one value
        if (not self.single_parameter_changes) & (not self.all_parameter_combinations):
            unique_param_lengths = set(simulation_parameter_lengths)
            param_lengths = unique_param_lengths - set([1])
            if len(param_lengths) > 1:
                raise ValueError("If single_parameter_changes is False, then each "
                                 "parameter can either have one value defined "
                                 "or as many values defined as all other "
                                 "parameters that have more than one value "
                                 "defined. However, at least one parameter "
                                 "has a different number of values defined.")

        if not self.single_parameter_changes:
            simulation_parameter_lengths = [nb
                                              for nb
                                              in simulation_parameter_lengths
                                              if nb > 1]
            if len(simulation_parameter_lengths) == 0:
                simulation_parameter_lengths = [1]
            else:
                simulation_parameter_lengths = [simulation_parameter_lengths[0]]

        elif not self.all_parameter_combinations:
            # if not all parameter combinations should be used
            # use the first parameter value as the standard and the other as single
            # changes that should be done (while leaving the other parameters
            # undchanged) - therefore for each simulation only one parameter is
            # changed from the standard set of parameter values

            # the number of simulations is 1 (for all standard parameter values)
            # plus the number of parameters beyond the standard parameter values
            # (which is the number of parameter values minus 1 for each
            # parameter)
            simulation_parameter_lengths = (np.sum(simulation_parameter_lengths)
                                            - len(simulation_parameter_lengths)
                                            + 1)
            simulation_parameter_lengths = [simulation_parameter_lengths]

        if self.dynamically_increase_nb_objects | (self.max_number_objects is None):
            # if the number of objects allowed in the simulation should be
            # dynamically increased, first check the maximum number of objects
            # that the simulation starts with
            max_number_objects = self._get_initial_max_nb_objects()
            # add the nb_objects_added_per_step to the maximum number of objects
            # to obtain the starting simulation array size
            self.max_number_objects = (max_number_objects +
                                       nb_objects_added_per_step)

        return simulation_parameter_lengths

    def _save_all_metadata(self, nb_simulations, max_number_objects):
        if self.simulations_summary_path is not None:
            summary_script_name = "simulations_summary.csv"
            simulations_summary_path = os.path.join(self.simulations_summary_path,
                                                    summary_script_name)
            if os.path.exists(simulations_summary_path):
                simulations_summary = pd.read_csv(simulations_summary_path,
                                                  index_col=0)
            else:
                simulations_summary = pd.DataFrame()

        # save metadata of simulations
        metadata = pd.DataFrame()
        folder_link = ('=HYPERLINK("file://' + self.data_folder +
                              '","'+os.path.basename(self.data_folder)+'")')
        metadata["folder"] = [folder_link]
        for parameter in self.parameters:
            metadata[parameter.name] = str(list(np.unique(parameter.values)))
        metadata["all_param_combinations"] = str(self.all_parameter_combinations)
        metadata["nb_of_states"] = [len(self.states)]
        metadata["states"] = [str([state.name for state in self.states])]
        metadata["nb_of_transitions"] = [len(self.transitions)]
        metadata["transitions"] = [str([transition.name
                                   for transition in self.transitions])]
        metadata["nb_of_properties"] = [len(self.properties)]
        metadata["properties"] = [str([property.name
                                       for property in self.properties])]
        metadata["nb_of_actions"] = [len(self.actions)]
        metadata["actions"] = [str([action.name for action in self.actions])]
        for property in self.properties:
            if hasattr(property._max_value, "cpu"):
               property._max_value = property._max_value.cpu()
            if hasattr(property._min_value, "cpu"):
               property._max_value = property._min_value.cpu()
            metadata[property.name+"_max"] = [str(property._max_value)]
            metadata[property.name+"_min"] = [str(property._min_value)]
        for name, data_extractor in self.data_extractions.items():
            metadata["data_"+name+"_resolution"] = [data_extractor.resolution]
            metadata["data_"+name+"_operation"] = [data_extractor.operation_name]
            metadata["data_"+name+"_state_groups"] = [data_extractor.state_groups]
        metadata["nb_simulations"] = [nb_simulations]
        metadata["max_number_objects"] = [max_number_objects]
        metadata["time_resolution"] = self.time_resolution
        metadata["min_time"] = self.min_time

        file_path = os.path.join(self.data_folder, "metadata.csv")
        metadata.to_csv(file_path)
        metadata = pd.read_csv(file_path)
        if self.simulations_summary_path is not None:
            for column in metadata.columns:
                if column.startswith("Unnamed"):
                    metadata = metadata.drop(column, axis=1)
                    continue

                if column not in simulations_summary.columns:
                    simulations_summary[column] = ""

            for column in simulations_summary.columns:
                if column not in metadata.columns:
                    metadata[column] = ""

            folder_in_summary = simulations_summary["folder"] == folder_link
            # prevent creating multiple entries for the same folder
            if len(simulations_summary.loc[folder_in_summary]) > 0:
                for column in metadata.columns:
                    val = metadata.iloc[0][column]
                    simulations_summary.loc[folder_in_summary, column] = [val]
                # print(len(metadata.iloc[0].values))
                # print(simulations_summary.loc[simulations_summary["folder"] ==
                #                         folder_link].values)
                # print(len(simulations_summary.loc[simulations_summary["folder"] ==
                #                         folder_link].values[0]))
                # simulations_summary.loc[simulations_summary["folder"] ==
                #                         folder_link] = metadata.iloc[0].values
            else:
                # if the folder is not present already, add it
                simulations_summary = pd.concat([simulations_summary, metadata])

            simulations_summary.to_csv(simulations_summary_path)

        print("Saving simulation object...")
        # pickle the entire simulation object
        file_path = os.path.join(self.data_folder, "SSA.pkl")
        with open(file_path, "wb") as file:
            dill.dump(self, file, fix_imports=False)

        if self.script_path is not None:
            if not os.path.exists(self.script_path):
                raise ValueError(f"The provided script path {self.script_path}"
                                 f" does not exist. The easiest way to supply "
                                 f"the correct path to the simulation script "
                                 f"is to use the __file__ variable.")
            script_file_name = os.path.basename(self.script_path)
            experiment = os.path.basename(self.data_folder)
            script_file_name = script_file_name.replace(".py", "_"+experiment+".py")
            new_script_path = os.path.join(self.data_folder, script_file_name)
            shutil.copy(self.script_path, new_script_path)

    @staticmethod
    def _get_number_of_cuda_cores():
        cc_cores_per_SM_dict = {
            (2, 0): 32,
            (2, 1): 48,
            (3, 0): 192,
            (3, 5): 192,
            (3, 7): 192,
            (5, 0): 128,
            (5, 2): 128,
            (6, 0): 64,
            (6, 1): 128,
            (7, 0): 64,
            (7, 5): 64,
            (8, 0): 64,
            (8, 6): 128,
            (8, 9): 128,
            (9, 0): 128
        }

        device = cuda.get_current_device()
        nb_SM = device.MULTIPROCESSOR_COUNT
        nb_cc_cores = cc_cores_per_SM_dict[device.compute_capability]
        return nb_SM, nb_cc_cores

    def _get_random_number_func(self, nb_parameter_combinations, nb_simulations, seed):
        # get way to get unique number for each combination of simulation
        # and parameter combination
        parameters_log10 = math.log10(nb_parameter_combinations)
        simulations_log10 = math.log10(nb_simulations)
        if parameters_log10 < simulations_log10:
            simulation_factor = int(10**(math.floor(parameters_log10) + 1))
            parameter_factor = 1
        else:
            simulation_factor = 1
            parameter_factor = int(10**(math.floor(simulations_log10) + 1))

        nb_states = ((nb_parameter_combinations+1) * parameter_factor +
                     (nb_simulations+1) * simulation_factor)

        random_number_func = numba.cuda.random.create_xoroshiro128p_states
        rng_states = random_number_func(int(nb_states), seed=int(seed))
        return rng_states, simulation_factor, parameter_factor

    def _empty_buffers(self):
        for property in self.properties:
            property.array_buffer = []

    def _extract_data(self):
        all_data = {}
        for data_name, data_extraction in self.data_extractions.items():
            all_data[data_name] = data_extraction.extract(self)
        self._empty_buffers()
        gc.collect
        return all_data

    def _add_objects_to_full_tensor(self):
        """
        Check whether there is a simulation in which all object positions
        are occupied. If there is, add room for more objects by increasing
        the array size for all arrays with object information.

        Returns: None

        """

        # empty cache whenever the tensor size was changed
        # print("\n EMPTY CACHE!")
        torch.cuda.empty_cache()

        #max_pos_with_object = positions_object[:,0].max()
        #if (max_pos_with_object < (self.object_states.shape[0] - 1)):
        #    return None

        self._simulation_array_size[0] += self.nb_objects_added_per_step
        self.max_number_objects += self.nb_objects_added_per_step
        # if that is the case, increase size of all arrays including object
        # information
        zero_array_float_to_add = torch.full((self.nb_objects_added_per_step,
                                               *self.object_states.shape[1:]),
                                             math.nan, dtype=torch.float)
        for property in self.properties:
            property.array = torch.cat((property.array,
                                        zero_array_float_to_add))

        zero_int_array_to_add = torch.zeros((self.nb_objects_added_per_step,
                                             *self.object_states.shape[1:]),
                                            dtype=torch.int8)
        self.object_states = torch.cat((self.object_states,
                                        zero_int_array_to_add))

        view_array = [-1] + [1] * (len(self._simulation_array_size) - 1)
        self.index_array = torch.linspace(1, self.max_number_objects,
                                          self.max_number_objects,
                                          dtype=torch.int16).view(*view_array)
        return None


    def get_tensor_memory(self):
        total_memory = 0
        for gc_object in gc.get_objects():
            try:
                if (hasattr(gc_object, "element_size") &
                        hasattr(gc_object, "nelement")):
                    try:
                        size = (gc_object.element_size() *
                                gc_object.nelement()/1024/1024)
                        total_memory += size
                        if size > 5:
                            print(type(gc_object),
                                  gc_object.shape, gc_object.dtype, size)
                    except:
                        continue
            except:
                continue
        print(total_memory)

    def _get_initial_max_nb_objects(self):
        # first sum all number, to get the total number of object with states
        # (after all states are assigned)
        max_number_objects_with_state = 0
        for state in self.states:
            if state.initial_condition is None:
                continue
            max_number_objects_with_state += np.max(state.initial_condition)

        return max_number_objects_with_state.item()

    def _initialize_parameter_arrays(self, all_simulation_parameters,
                                     simulation_parameter_lengths,
                                     single_parameter_changes,
                                     all_parameter_combinations,
                                     nb_timepoints):
        # go through all model parameters and expand array to simulation
        # specific size
        all_dimensions = [dim for dim
                          in range(len(simulation_parameter_lengths) + 2)]
        nb_previous_params = 0
        for dimension, model_parameters in enumerate(all_simulation_parameters):
            all_timepoints_param_vals = model_parameters.values
            param_switch_timepoints = model_parameters.switch_timepoints
            if param_switch_timepoints is not None:
                param_switch_timepoints = np.floor(param_switch_timepoints /
                                                   self.time_resolution).astype(np.int32)
            all_timepoints_array = None
            # go through parameter values for all timepoints
            # each sublist of parameter values represents the values until
            # a defined switchpoint. The index of the parameter values indicates
            # the index of the switch_timepoints at which parameter values
            # are switched to the next index (next group). For the last group
            # there is no switch timepoint defined, since these parameter
            # values will be active until the last timepoint
            last_switch_timepoint = 0
            for (param_timepoint_nb,
                 param_vals) in enumerate(all_timepoints_param_vals):
                # if param_switch_timepoints is defined
                if param_switch_timepoints is not None:
                    if param_timepoint_nb < len(param_switch_timepoints):
                        switch_timepoint = param_switch_timepoints[param_timepoint_nb]
                    else:
                        switch_timepoint = nb_timepoints
                else:
                    switch_timepoint = nb_timepoints

                if not single_parameter_changes:
                    if len(param_vals) == 1:
                        array = param_vals.repeat((simulation_parameter_lengths))
                    else:
                        array = model_parameters.values
                    array = torch.Tensor(array)
                    # array = array.expand(1, *self._simulation_array_size[1:])
                    # model_parameters.value_array = array

                elif all_parameter_combinations:
                    array_dimension = dimension + 2
                    expand_dimensions = copy.copy(all_dimensions)
                    expand_dimensions.remove(array_dimension)
                    # expand dimensions of parameter values to simulation array
                    array = np.expand_dims(param_vals,
                                           expand_dimensions)
                    # save array in object, therefore also change objects saved in
                    # self.transitions and self.actions
                    array = torch.Tensor(array)
                else:
                    standard_value = param_vals[0]
                    standard_value = torch.Tensor([standard_value])
                    # use standard value across entire array
                    array = standard_value.repeat((simulation_parameter_lengths))
                    # then get the position/s at which the values should be changed
                    # which is starting at the number of previously changed
                    # positions (for other parameters)
                    # the first position should be all the standard values without
                    # any change
                    # the following positions should be the single parameter changes
                    # therefore, the number of additional parameters is the total
                    # number minus 1
                    nb_add_params = len(model_parameters.values) - 1
                    if len(model_parameters.values) > 1:
                        start = nb_previous_params + 1
                        end = start + nb_add_params
                        array[start:end] = model_parameters.values[1:]
                    nb_previous_params += nb_add_params
                if all_timepoints_array is None:
                    all_timepoints_array = np.zeros((nb_timepoints,
                                                     *array.shape))
                array = np.expand_dims(array, axis=0)
                # use current array until the timepoint before the
                # switch_timepoint
                all_timepoints_array[last_switch_timepoint:
                                     switch_timepoint] = array
                # update last switch timepoint
                last_switch_timepoint = switch_timepoint

            # array = array.expand(1,*self._simulation_array_size[1:])
            # reshape to have all different parameter values in one axis
            model_parameters.value_array = all_timepoints_array
            # model_parameters.value_array = np.array(array.reshape((array.shape[1],
            #                                                       -1)))
        return None

    def _initialize_object_states(self, param_shape_batch):
        # initial object states, also using defined initial condition of
        # number of objects starting in each state
        self.object_states = torch.zeros((self._simulation_array_size[0],
                                         *param_shape_batch),
                                         dtype=torch.int8)

        # keep track which objects already have a state set
        # to assign states to the correct positions
        # due to combinations of different number of objects for different
        # states, keep track of already assigned objects for each state
        object_state_shape = self.object_states.shape
        nb_objects_with_states = torch.zeros((1,*object_state_shape[1:]),
                                             dtype=torch.int16)
        # first sum all number, to get the total number of object with states
        # (after all states are assigned)
        for state in self.states:
            if state.initial_condition is None:
                continue
            # expand the initial condition array, to add them up
            initial_cond = state.initial_condition
            expanded_array = torch.ShortTensor(initial_cond)
            expanded_array = expanded_array.expand((1,*object_state_shape[1:]))
            nb_objects_with_states += expanded_array

        # from all initial conditions, if the
        if ((torch.max(nb_objects_with_states) > self.object_states.shape[0]) &
                (self.ignore_errors == False)):
            raise ValueError(f"Initial conditions for states "
                             f"implied more objects with a state "
                             f"than the defined maximum number of "
                             f"objects. After state {state.name} the "
                             f"total number of objects would be "
                             f"{torch.max(nb_objects_with_states)} "
                             f"which is more than the maximum allowed "
                             f"number {self.max_number_objects}.")

        # then subtract assigned number of objects at each state
        # so that the number of objects that are assigned gets lower
        # with each state, thereby preventing overriding already signed states
        # (through keeping threshold for the index array lower)
        for state in self.states:
            if state.initial_condition is None:
                continue
            # expand the initial condition array, to add them up and get number
            # of assigned objects for each simulation
            initial_cond = state.initial_condition
            expanded_array = torch.ShortTensor(initial_cond)
            expanded_array = expanded_array.expand((1,*object_state_shape[1:]))

            object_state_mask = (self.index_array.expand(
                (self._simulation_array_size[0],*param_shape_batch))
                                 <= nb_objects_with_states)

            self.object_states[object_state_mask] = state.number
            # subtract the number of objects for the current state
            # thereby defining which number of objects will not be overwritten
            # by next state and thereby stay in assigned state
            nb_objects_with_states -= expanded_array

        return None

    def _initialize_object_properties(self, param_shape_batch):
        # create tensor for each object property
        # so that each object can have a value for each property
        # also respect initial condition, if defined
        object_state_mask = self.object_states > 0
        nb_objects_with_states = torch.count_nonzero(object_state_mask)
        for object_property in self.properties:
            random_property_vals = False
            object_property.array = torch.zeros((self._simulation_array_size[0],
                                                 *param_shape_batch),
                                                dtype=torch.float32)
            object_property.array[:] = math.nan
            initial_cond = object_property.initial_condition

            if ((initial_cond is not None) &
                    (type(initial_cond) == type(self.__init__))):
                # if initial condition is defined and a function,
                # get values from function
                property_values = initial_cond(nb_objects_with_states)
            elif ((initial_cond is not None) &
                  (type(initial_cond) == list)):
                # if initial condition is a string == "random
                # then random numbers from min to max val should be generated
                if len(initial_cond) == 2:
                    min_value = initial_cond[0]
                    max_value = initial_cond[1]
                    random_property_vals = True
                else:
                    raise ValueError("For initial condition for object "
                                     "properties a string only 'random' is "
                                     "implemented. For the property"
                                     f"{object_property.name} {initial_cond}"
                                     f" was used instead.")
            elif (initial_cond is not None):
                # otherwise if initial condition is defined, must be number
                # that should be used for all objects initially
                property_values = initial_cond

            elif (type(object_property.start_value) == list):
                # if no initial cond is defined, use the start value instead.
                # if start_value is a list, property values will be
                # random number between first and second element
                min_value = object_property.start_value[0]
                max_value = object_property.start_value[1]
                random_property_vals = True
            else:
                # otherwise, start value is a single number
                property_values = object_property.start_value

            if random_property_vals:
                get_property_vals = self._get_random_poperty_values
                property_values = get_property_vals(min_value, max_value,
                                                    nb_objects_with_states)
                property_values = property_values.to(torch.float32)
                # if there is just one property value (its not a tensor)
                # use another way of assinging values
                # masked_scatter has advantage of much smaller memory footprint
                # but when just assigning one value, memory footprint using
                # mask directly is small

            if type(property_values) != type(object_property.array):
                object_property.array[object_state_mask] = property_values
            else:
                object_property.array = object_property.array.masked_scatter(object_state_mask,
                                                                             property_values)

        return None

    def _get_random_poperty_values(self, min_value,
                                   max_value, nb_objects):
        # scale random number from min_value to max_value
        property_values = (torch.rand((nb_objects), dtype=torch.float) *
                           (max_value - min_value) + min_value)
        return property_values


    def _get_finished_simulation(self, dim_list, iteration_nb):
        all_finished_sim_positions = []
        for dim in range(1,len(self.times.shape)):
            new_dim_list = copy.copy(dim_list)
            new_dim_list.remove(dim)
            min_sim_time_param_value = torch.amin(self.times,
                                                  dim=tuple(new_dim_list))
            finished_sim_positions = torch.where(min_sim_time_param_value >=
                                                 self.min_time)[0]

            all_finished_sim_positions.append(finished_sim_positions)
            # save the specific positions that are removed from array
            # in a dataframe
            # with columns (iteration_nb, dimension, position)
            # When all simulations are done, save the dataframe as .feather
            for finished_sim_position in finished_sim_positions:
                nb_removed_positions = len(self.all_removed_positions)
                # user iteration_nb + 1 since it will only be removed for
                # the next iteration - otherwise the final step in the
                # simulation would be excluded
                new_removed_position = pd.DataFrame({"iteration_nb":
                                                         iteration_nb+1,
                                                     "dimension": dim,
                                                     "position":
                                                         finished_sim_position.item()},
                                                    index=
                                                    [nb_removed_positions])
                concat_list = [self.all_removed_positions, new_removed_position]
                self.all_removed_positions = pd.concat(concat_list)
        return all_finished_sim_positions

    def _remove_finished_positions_from_all_arrays(self,
                                                   all_finished_sim_positions):
        # Remove data from finished simulation positions
        # create list with lists of all shape positions
        all_dim_list = [list(range(shape)) for shape in self.times.shape]
        positions_removed = False
        for dim, dim_list in enumerate(all_dim_list[1:]):
            finished_sim_positions = all_finished_sim_positions[dim]
            dim += 1
            if len(finished_sim_positions) == 0:
                continue

            positions_removed = True

            for finished_sim_position in finished_sim_positions:
                dim_list.remove(finished_sim_position)
            # make all arrays smaller
            dim_list_tensor = torch.IntTensor(dim_list)
            self.times = torch.index_select(self.times, dim=dim,
                                            index=dim_list_tensor)
            for property in self.properties:
                property.array = torch.index_select(property.array,
                                                    dim=dim,
                                                    index=dim_list_tensor)
            self.object_states = torch.index_select(self.object_states,
                                                    dim=dim,
                                                    index=dim_list_tensor)

            for sim_parameter in self._all_simulation_parameters:
                sim_parameter.value_array = torch.index_select(sim_parameter.value_array,
                                                             dim=dim+1,
                                                             index=dim_list_tensor)

        # clear cache if a position was removed and therefore tensor size changed
        if positions_removed:
            torch.cuda.empty_cache()

        self._simulation_array_size = [self._simulation_array_size[0],
                                       *self.times.shape[1:]]

        return None

    def _free_gpu_memory(self):
        # check if cache should be emptied

        # check how much space is free on the GPU (if executed on GPU)
        start = time.time()
        #print(6.1, time.time() - start)
        start = time.time()
        total_memory = torch.cuda.get_device_properties(0).total_memory
        reserved_memory = torch.cuda.memory_reserved(0)
        free_memory = (total_memory - reserved_memory)

        # get maximum memory used in current iteration
        iteration_memory_used = torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()

        # print("RESERVED:", reserved_memory/1024/1024/1024,
        # "   ALLOCATED:", torch.cuda.memory_allocated(0)/1024/1024/1024,
        # "   ITERATION USED:", iteration_memory_used/1024/1024/1024,
        # "   FREE RESERVED:", free_memory/1024/1024/1024,
        # "   FREE ALLOCATED:",(total_memory-torch.cuda.memory_allocated(0))/1024/1024/1024)

        #print(6.4, time.time() - start)
        #print("\nITERATION NB: ", iteration_nb)
        # if 2x space free on GPU than would be needed for data, keep on GPU

    def _add_data_to_buffer(self):
        # add buffer for all data
        self.object_states_buffer.append(self.object_states.unsqueeze(0).clone())
        for property in self.properties:
            property.array_buffer.append(property.array.unsqueeze(0).clone())
        self.times_buffer.append(self.times.unsqueeze(0).clone())

    def _save_times_and_object_states(self, iteration_nb):

        # times_array = torch.concat(self.times_buffer)
        times_array = self.times
        file_path = os.path.join(self.data_folder,
                                    "times_" + str(iteration_nb) + ".pt")
        torch.save(times_array, file_path)

        self.times_buffer = []

        if self.save_states:
            # object_state_array = torch.concat(self.object_states_buffer)
            object_state_array = self.object_states
            file_path = os.path.join(self.data_folder,
                                        "states_" + str(iteration_nb) + ".pt")
            torch.save(object_state_array, file_path)
        self.object_states_buffer = []

    def _concat_data_from_buffer(self):
        all_data = {}
        for data in self.data_buffer:
            for data_name, sub_data in data.items():
                if data_name not in all_data:
                    all_data[data_name] = {}
                for keyword, data_values in sub_data.items():
                    if keyword not in all_data[data_name]:
                        all_data[data_name][keyword] = []
                    all_data[data_name][keyword].append(data_values)
        self.data_buffer = []

        all_concat_data = {}
        for data_name, sub_data in all_data.items():
            all_concat_data[data_name] = {}
            for keyword, data_values in sub_data.items():
                concat_data = torch.concat(all_data[data_name][keyword])
                all_concat_data[data_name][keyword] = concat_data
        return all_concat_data

    def _save_data(self, data_dict, iteration_nb):

        data_dict_cpu = {}
        for data_name, data in data_dict.items():
            data_dict_cpu[data_name] = {}
            for keyword, data_array in data.items():
                data_dict_cpu[data_name][keyword] = data_array.cpu()

        # save data to hard drive
        for data_name, data in data_dict.items():
            for keyword, data_array in data.items():
                sub_data_folder = os.path.join(self.data_folder,
                                               "_data_"+data_name)
                if not os.path.exists(sub_data_folder):
                    os.mkdir(sub_data_folder)
                file_path = os.path.join(sub_data_folder,
                                         data_name+"_"+keyword+"_"+
                                         str(iteration_nb)+".pt")
                torch.save(data_array, file_path)

        return None

    def _save_simulation_parameters(self):
        for parameter in self.parameters:
            file_name = "param_"+parameter.name+".pt"
            torch.save(parameter.value_array, os.path.join(self.data_folder,
                                                           file_name))
        return None
