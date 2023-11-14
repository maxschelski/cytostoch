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
                 object_removal=None, name="", device="GPU",
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
        self.name = name

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
              ignore_errors=False, print_update_time_step=1,
               nb_objects_added_per_step=10,
              max_iters_no_data_extraction=2000,
               dynamically_increase_nb_objects=False,
              remove_finished_sims=False, save_states=False,
              use_assertion_checks=True, reset_folder=True):
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
        Returns:

        """
        # turn off autograd function to reduce memory overhead of pytorch
        # and reduce backend processes
        with torch.no_grad():
            self._start(nb_simulations, min_time, data_extractions, data_folder,
                        time_resolution, save_initial_state,
                        max_number_objects, ignore_errors,
                        print_update_time_step,
                        nb_objects_added_per_step,
                        max_iters_no_data_extraction,
                        dynamically_increase_nb_objects,
                        remove_finished_sims, save_states,
                        use_assertion_checks, reset_folder)

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
            properties_array[property_nb] = property.array
        return properties_array

    def _get_property_min_max_val_arrays(self):
        nb_properties = len(self.properties)
        # if there is just one non-nan value in min_value
        # then this is the min_value
        # if there are multiple non-nan values in min_value
        # then these are property numbers, except
        # for the first value, which is threshold
        # and the second value, which is the operation
        # (-1 for subtracting property values,
        #  +1 for adding property values)
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
            max_nb_max_value_properties += 2
        if max_nb_min_value_properties > 0:
            max_nb_min_value_properties += 2

        property_min_values = np.full((nb_properties,
                                       max(1, max_nb_min_value_properties)),
                                      math.nan)
        property_max_values = np.full((nb_properties,
                                       max(1, max_nb_max_value_properties)),
                                      math.nan)

        for property_nb, property in enumerate(self.properties):
            min_value = property._min_value
            if type(min_value) in [float, int]:
                property_min_values[property_nb, 0] = min_value
            elif min_value is not None:
                property_min_values[property_nb, 0] = min_value.properties[
                    0].min_value
                operation = min_value._operation
                if operation == "same_dimension_forward":
                    property_min_values[property_nb, 1] = 1
                for (min_val_nb,
                     min_val_property) in enumerate(min_value.properties):
                    property_min_values[property_nb,
                                        min_val_nb + 2] = min_val_property.number

            max_value = property._max_value
            if type(max_value) in [float, int]:
                property_max_values[property_nb, 0] = max_value
            elif max_value is not None:
                property_max_values[property_nb, 0] = \
                    max_value.properties[0].max_value
                operation = max_value._operation
                if operation == "same_dimension_forward":
                    property_max_values[property_nb, 1] = 1
                for (min_val_nb,
                     min_val_property) in enumerate(max_value.properties):
                    property_max_values[property_nb,
                                        min_val_nb + 2] = min_val_property.number

        return property_min_values, property_max_values

    def _get_parameter_value_array(self):
        nb_parameters = len(self.parameters)
        parameter_value_shape = self.parameters[0].value_array.shape
        parameter_value_array = np.zeros((nb_parameters,
                                          *parameter_value_shape))
        for nb, parameter in enumerate(self.parameters):
            parameter_value_array[nb] = parameter.value_array
        return parameter_value_array

    def _get_transition_parameters(self):
        # - all transition rates
        nb_transitions = len(self.transitions)
        transition_parameters = np.zeros((nb_transitions))
        for nb, transition in enumerate(self.transitions):
            transition_parameters[nb] = transition.parameter.number
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

    def _get_transition_set_to_zero_properties(self):
        nb_transitions = len(self.transitions)
        max_nb_properties_set_to_zero = 0
        for nb, transition in enumerate(self.transitions):
            properties_set_to_zero = transition.properties_set_to_zero
            if properties_set_to_zero is None:
                continue
            max_nb_properties_set_to_zero = max(max_nb_properties_set_to_zero,
                                                len(properties_set_to_zero))

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
        # property value being larger then a threshold
        # and -1 for the combined property value being smaller then a threshold
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

    def _start(self, nb_simulations, min_time, data_extractions,data_folder,
               time_resolution, save_initial_state=False,
               max_number_objects=None, ignore_errors=False,
               print_update_time_step=1, nb_objects_added_per_step=10,
                max_iters_no_data_extraction=2000,
               dynamically_increase_nb_objects=False,
               remove_finished_sims=False, save_states=False,
               use_assertion_checks=True, reset_folder=True):
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
        Returns:
        """
        if reset_folder:
            if os.path.exists(data_folder):
                shutil.rmtree(data_folder)
                time.sleep(0.1)
            os.mkdir(data_folder)

        self.min_time = min_time
        self.data_extractions = data_extractions
        self.data_folder = data_folder
        self.ignore_errors = ignore_errors
        self.nb_objects_added_per_step = nb_objects_added_per_step
        self.dynamically_increase_nb_objects = dynamically_increase_nb_objects
        self.save_states = save_states
        self.use_assertion_checks = use_assertion_checks
        self.max_iters_no_data_extraction = max_iters_no_data_extraction
        self.remove_finished_sims = remove_finished_sims

        self.all_data = []
        self.all_times = []
        self.all_states = []
        self.all_iteration_nbs = []

        self.times_buffer = []
        self.object_states_buffer = []

        self.initial_memory_used = 0
        self.iteration_memory_used = 0

        self.all_removed_positions = pd.DataFrame()

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

        # create list with all transitions and then with all actions
        self._all_simulation_parameters = [*self.states, *self.parameters]

        # create list of length of all model parameters
        simulation_parameter_lengths = [len(parameters.values)
                                        for parameters
                                        in self._all_simulation_parameters]
        all_parameter_combinations = False
        # if not all parameter combinations should be used
        # use the first parameter value as the standard and the other as single
        # changes that should be done (while leaving the other parameters
        # undchanged) - therefore for each simulation only one parameter is
        # changed from the standard set of parameter values
        if not all_parameter_combinations:
            # the number of simulations is 1 (for all standard parameter values)
            # plus the number of parameters beyond the standard parameter values
            # (which is the number of parameter values minus 1 for each
            # parameter)
            simulation_parameter_lengths = (np.sum(simulation_parameter_lengths)
                                            - len(simulation_parameter_lengths)
                                            + 1)
            simulation_parameter_lengths = [simulation_parameter_lengths]

        if dynamically_increase_nb_objects | (max_number_objects is None):
            # if the number of objects allowed in the simulation should be
            # dynamically increased, first check the maximum number of objects
            # that the simulation starts with
            max_number_objects = self._get_initial_max_nb_objects()
            # add the nb_objects_added_per_step to the maximum number of objects
            # to obtain the starting simulation array size
            self.max_number_objects = (max_number_objects +
                                       nb_objects_added_per_step)
        else:
            self.max_number_objects = max_number_objects

        # array size contains for each combination of parameters to explore
        self._simulation_array_size = [self.max_number_objects, nb_simulations,
                                       *simulation_parameter_lengths]

        self._initialize_parameter_arrays(self._all_simulation_parameters,
                                          simulation_parameter_lengths,
                                          all_parameter_combinations)

        self.times = torch.zeros((1,*self._simulation_array_size[1:]))

        # create index array in which each entry has the value of the index of
        # the microtubule in the simulation, thereby multi-D operations
        # on single simulations can be executed
        # create array of index numbers of same shape as whole simulation array
        view_array = [-1] + [1] * (len(self._simulation_array_size) - 1)
        max_number_objects = self.max_number_objects

        self.index_array = torch.linspace(1, max_number_objects,
                                          max_number_objects,
                                          dtype=torch.int16).view(*view_array)

        self._initialize_object_states()

        self._initialize_object_properties()

        # self.get_tensor_memory()

        self._add_objects_to_full_tensor()

        # data = self.data_extraction.extract(self)
        #
        # if self.device.find("cuda") != -1:
        #     self._free_gpu_memory()
        #
        # self._save_data(data, 0)

        # reshape to have all different parameter values in one axis
        # with the first axis being the number of objects
        self.times = np.array(self.times.reshape((self.times.shape[0],
                                                  self.times.shape[1], -1)))

        self.object_states = self.object_states.reshape((self.object_states.shape[0],
                                                         self.object_states.shape[1],
                                                         -1))

        for property in self.properties:
            property.array = property.array.reshape((property.array.shape[0],
                                                     property.array.shape[1],
                                                     -1))
            property.array = np.array(property.array)

        # self._add_data_to_buffer()
        self._save_simulation_parameters()

        self.data_buffer = []
        self.last_data_extraction = 0
        self.last_data_saving = 0

        # to make run_iteration compatible with numba.njit(debug=True) (numba no python)
        # use numpy arrays for everything instead of python objects
        # create one array for

        # create arrays with the parameter size to be explored in one round

        self.object_states = np.array(self.object_states)

        property_start_values = self._get_property_start_val_arrays()

        properties_array = self._get_property_vals_array()

        (property_min_values,
         property_max_values) = self._get_property_min_max_val_arrays()

        parameter_value_array = self._get_parameter_value_array()

        transition_parameters = self._get_transition_parameters()
        all_transition_states = self._get_transition_state_arrays()
        get_set_zero_array = self._get_transition_set_to_zero_properties()
        all_transition_set_to_zero_properties = get_set_zero_array
        get_transfer_vals_array = self._get_transition_transferred_vals_array()
        all_transition_tranferred_vals = get_transfer_vals_array

        action_parameters = self._get_action_parameter_array()
        all_action_properties = self._get_action_properties_array()
        action_operation_array = self._get_action_operation_array()
        action_state_array = self._get_action_states_array()

        all_object_removal_properties = self.get_object_removal_property_array()
        object_removal_operations = self._get_object_removal_property_array()

        # -  current rates of all transitions (empty)
        nb_transitions = len(self.transitions)
        current_transition_rates = np.zeros((nb_transitions,
                                             *self.object_states.shape[1:]))
        # print(action_parameters, transition_parameters,
        #       current_parameter_values.shape,
        #       parameter_value_array.shape)

        # - all total transition rates (empty)
        total_rates = np.zeros(self.object_states.shape[1:])

        # - current reaction times (empty)
        reaction_times = np.zeros(self.object_states.shape[1:])

        current_transitions = np.zeros(self.object_states.shape[1:])

        all_transition_positions = np.zeros(self.object_states.shape[1:])

        # calculate number of objects for all states
        nb_objects_all_states = np.zeros((len(self.states),
                                          *self.object_states.shape[1:]))
        for state_nb, state in enumerate(self.states):
            nb_objects_all_states[state_nb] = np.sum(self.object_states ==
                                                     state.number, axis=0)

        # get number of timepoints
        nb_timepoints = math.ceil(self.min_time/time_resolution)
        if save_initial_state:
            nb_timepoints += 1

        # set arrays for values over time
        current_timepoint_array = np.zeros((self.object_states.shape[1:]))

        timepoint_array = np.full((nb_timepoints,
                                   *self.object_states.shape[1:]),
                                  math.nan)
        if save_initial_state:
            timepoint_array[0] = 0

        object_state_time_array = np.full((nb_timepoints,
                                           *self.object_states.shape),
                                          math.nan)
        if save_initial_state:
            object_state_time_array[0] = self.object_states

        nb_properties = len(self.properties)
        properties_time_array = np.full((nb_timepoints, nb_properties,
                                         *self.object_states.shape),
                                        math.nan)

        if save_initial_state:
            for property_nb, property in enumerate(self.properties):
                properties_time_array[0, property_nb] = property.array

        # current_min_time = np.min(self.times)
        # if current_min_time >= min_time:
        #     break
        convert_array = np.ascontiguousarray

        current_timepoint_array = convert_array(current_timepoint_array)
        timepoint_array = convert_array(timepoint_array)
        object_state_time_array = convert_array(object_state_time_array)
        properties_time_array = convert_array(properties_time_array)
        time_resolution = convert_array(time_resolution)

        seed = 42

        device = "gpu"

        nb_simulations = self.object_states.shape[1]
        nb_parameter_combinations = self.object_states.shape[2]

        if self.device.find("cuda") != -1:
            simulation_numba._decorate_all_functions_for_gpu()

            start = time.time()

            to_cuda = cuda.to_device

            # get number of cuda stream managers and cores per stream manager
            nb_SM, nb_cc = self._get_number_of_cuda_cores()

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
            rng_states = random_number_func(nb_states, seed=seed)

            # get free GPU memory if device is cuda, to do calculations in batches
            # that fit on the GPU
            (free_memory,
             total_memory) = numba.cuda.current_context().get_memory_info()

            # get size of all arrays that were created
            # the size depends on

            # get size of object states
            size_states = (self.object_states.size *
                           self.object_states.itemsize)

            # get size of properties times the number of properties+
            size_properties = (len(self.properties) *
                               self.properties[0].array.size *
                               self.properties[0].array.itemsize)

            total_size = (nb_timepoints+1) * (size_states + size_properties)

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
            total_nb_simulations = (self.object_states.shape[1] *
                                    self.object_states.shape[2])
            nb_core_batches = int(math.ceil(total_nb_simulations / (nb_SM *
                                                                    nb_cc)))

            nb_batches = max(nb_memory_batches, 1)

            parameter_combinations_per_batch = int(math.floor(
                (nb_parameter_combinations / nb_batches)))

            object_states_batches = None
            properties_batches_array = None
            times_batches = None

            for batch_nb in range(nb_batches):
                start_parameter_comb = (batch_nb *
                                        parameter_combinations_per_batch)
                end_parameter_comb = (start_parameter_comb +
                                      parameter_combinations_per_batch)
                param_slice = slice(start_parameter_comb, end_parameter_comb)

                current_timepoint_array_batch = cuda.to_device(
                    convert_array(current_timepoint_array[:,param_slice]))
                timepoint_array_batch = cuda.to_device(
                    convert_array(timepoint_array[:,:,param_slice]))
                object_state_time_array_batch = cuda.to_device(
                    convert_array(object_state_time_array[:,:,:,param_slice]))
                properties_time_array_batch = cuda.to_device(
                    convert_array(properties_time_array[:,:,:,:,param_slice]))
                time_resolution = cuda.to_device(time_resolution)


                print("Starting simulation batch...")
                sim = simulation_numba._execute_simulation_gpu
                object_states_batch = self.object_states[:,:,param_slice]
                property_array_batch = properties_array[:,:,:,param_slice]
                times_batch = self.times[:,:,param_slice]
                param_val_array_batch = parameter_value_array[:,:,param_slice]
                current_trans_rates = current_transition_rates[:,:,param_slice]
                nb_obj_all_states_batch = nb_objects_all_states[:,:,param_slice]
                total_rates_batch = total_rates[:,param_slice]
                reaction_times_batch = reaction_times[:,param_slice]
                current_transitions_batch = current_transitions[:,param_slice]
                all_trans_pos_batch = all_transition_positions[:,param_slice]

                sim[nb_SM,
                 nb_cc](to_cuda(convert_array(object_states_batch)),
                            to_cuda(convert_array(property_array_batch)),
                            to_cuda(convert_array(times_batch)),
                            nb_simulations, parameter_combinations_per_batch,
                            to_cuda(convert_array(param_val_array_batch)),
                            to_cuda(convert_array(transition_parameters)),
                            to_cuda(convert_array(all_transition_states)),
                            to_cuda(convert_array(action_parameters)),
                            to_cuda(convert_array(action_state_array)),
                            to_cuda(convert_array(all_action_properties)),
                            to_cuda(convert_array(action_operation_array)),
                            to_cuda(convert_array(current_trans_rates)),
                            to_cuda(convert_array(property_start_values)),
                            to_cuda(convert_array(property_min_values)),
                            to_cuda(convert_array(property_max_values)),
                            to_cuda(convert_array(
                                all_transition_tranferred_vals)),
                            to_cuda(convert_array(
                                all_transition_set_to_zero_properties)),
                            to_cuda(convert_array(
                                all_object_removal_properties)),
                            to_cuda(convert_array(object_removal_operations)),

                            to_cuda(convert_array(nb_obj_all_states_batch)),
                            to_cuda(convert_array(total_rates_batch)),
                            to_cuda(convert_array(reaction_times_batch)),
                            to_cuda(convert_array(current_transitions_batch)),
                            to_cuda(convert_array(all_trans_pos_batch)),

                           current_timepoint_array_batch,
                           timepoint_array_batch,
                           object_state_time_array_batch,
                           properties_time_array_batch,
                           time_resolution, self.min_time, save_initial_state,
                           seed, rng_states, simulation_factor, parameter_factor
                             )

                print(time.time() - start)

                numba.cuda.synchronize()
                object_state_batch = torch.Tensor(
                    object_state_time_array_batch.copy_to_host())
                property_array_batch = torch.Tensor(
                    properties_time_array_batch.copy_to_host())
                times_batch = torch.Tensor(timepoint_array_batch.copy_to_host())

                if object_states_batches is None:
                    object_states_batches = object_state_batch
                else:
                    object_states_batches = torch.concat((object_states_batches,
                                                         object_state_batch),
                                                         axis=-1)
                if properties_batches_array is None:
                    properties_batches_array = property_array_batch
                else:
                    properties_batches_array = torch.concat((properties_batches_array,
                                                            property_array_batch),
                                                            axis=-1)
                if times_batches is None:
                    times_batches = times_batch
                else:
                    times_batches = torch.concat((times_batches, times_batch),
                                                 axis=-1)

                del current_timepoint_array_batch
                del timepoint_array_batch
                del object_state_time_array_batch
                del properties_time_array_batch
                cuda.current_context().memory_manager.deallocations.clear()

            self.times = (times_batches * time_resolution.copy_to_host())
            self.times = self.times.unsqueeze(1).to(self.device)

            for property_nb, property in enumerate(self.properties):
                property.array = properties_batches_array[:, property_nb]
                property.array = property.array.to(self.device)
            self.object_states = object_states_batches.to(self.device)

        else:
            simulation_numba._decorate_all_functions_for_cpu()
            print("Starting simulation...")
            start = time.time()
            _execute_sim = simulation_numba._execute_simulation_cpu
            _execute_sim(convert_array(self.object_states),
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
                        convert_array(property_min_values),
                        convert_array(property_max_values),
                        convert_array(all_transition_tranferred_vals),
                        convert_array(all_transition_set_to_zero_properties),
                        convert_array(all_object_removal_properties),
                        convert_array(object_removal_operations),

                        convert_array(nb_objects_all_states),
                        convert_array(total_rates),
                        convert_array(reaction_times),
                        convert_array(current_transitions),
                        convert_array(all_transition_positions),

                        current_timepoint_array, timepoint_array,
                        object_state_time_array, properties_time_array,
                        time_resolution, self.min_time, save_initial_state,seed
                        )
            print(time.time() - start)

            self.object_states = torch.Tensor(np.copy(object_state_time_array))

            for property_nb, property in enumerate(self.properties):
                property.array = torch.Tensor(np.copy(
                    properties_time_array[:, property_nb]))

            self.times = (torch.Tensor(np.copy(timepoint_array)).unsqueeze(1)
                          * time_resolution)

        all_data = {}
        for data_name, data_extraction in self.data_extractions.items():
            start = time.time()
            all_data[data_name] = data_extraction.extract(self)
            print(data_name, "extraction time: ", time.time() - start)
            if data_name.startswith("local_density"):
                for sub_name, sub_data in all_data[data_name].items():
                    if sub_data.sum() == 0:
                        continue
                    plt.figure()
                    plt.plot(torch.mean(sub_data[-1].cpu(),dim=(1)))
                    plt.title(sub_name)
                    plt.ylim(0,plt.ylim()[1])
                    # plt.figure()
                    # plt.plot(torch.mean(sub_data[-1],dim=(1)))
                    # plt.ylim(0,3.5)
            else:
                for sub_name, sub_data in all_data[data_name].items():
                    if sub_name.find("mean") == -1:
                        continue
                    # if sub_name.find("inside") == -1:
                    #     continue
                    mean = np.nanmean(sub_data[-1].cpu())
                    if mean == 0:
                        continue
                    if np.isnan(mean):
                        continue
                    print(sub_name, mean)
        self._save_times_and_object_states(0)
        self._save_data(all_data, 0)

    def _get_number_of_cuda_cores(self):
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
                                     all_parameter_combinations):
        # go through all model parameters and expand array to simulation
        # specific size
        all_dimensions = [dim for dim
                          in range(len(simulation_parameter_lengths) + 2)]

        nb_previous_params = 0
        for dimension, model_parameters in enumerate(all_simulation_parameters):

            if all_parameter_combinations:
                array_dimension = dimension + 2
                expand_dimensions = copy.copy(all_dimensions)
                expand_dimensions.remove(array_dimension)
                # expand dimensions of parameter values to simulation array
                array = np.expand_dims(model_parameters.values,
                                       expand_dimensions)
                # save array in object, therefore also change objects saved in
                # self.transitions and self.actions
                array = torch.HalfTensor(array)
            else:
                standard_value = model_parameters.values[0]
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
            array = array.expand(1,*self._simulation_array_size[1:])
            # reshape to have all different parameter values in one axis
            model_parameters.value_array = np.array(array.reshape((array.shape[1],
                                                                  -1)))
        return None

    def _initialize_object_states(self):
        # initial object states, also using defined initial condition of
        # number of objects starting in each state
        self.object_states = torch.zeros(self._simulation_array_size,
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
                self._simulation_array_size) <= nb_objects_with_states)

            self.object_states[object_state_mask] = state.number
            # subtract the number of objects for the current state
            # thereby defining which number of objects will not be overwritten
            # by next state and thereby stay in assigned state
            nb_objects_with_states -= expanded_array

        return None

    def _initialize_object_properties(self):
        # create tensor for each object property
        # so that each object can have a value for each property
        # also respect initial condition, if defined
        object_state_mask = self.object_states > 0
        nb_objects_with_states = torch.count_nonzero(object_state_mask)
        for object_property in self.properties:
            random_property_vals = False
            object_property.array = torch.zeros(self._simulation_array_size,
                                                dtype=torch.float)
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
                                                             dim=dim,
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
        for parameter in self._all_simulation_parameters:
            file_name = "param_"+parameter.name+".pt"
            torch.save(parameter.value_array, os.path.join(self.data_folder,
                                                           file_name))
        return None
