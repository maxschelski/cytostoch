# -*- coding: utf-8 -*-
"""
Write stochastic model of microtubules in neurite
"""

import torch
import copy
import numpy as np
import pandas as pd
import time
import sys
import gc
import os
import psutil

from . import analyzer

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
- action state transition class (includes one transition, rates 
                                 (rate can be supplied as lifetime))
"""

class tRSSA():

    def __init__(self):
        pass

    def start(self, time_step_size_simulation, time_depency_step_size=0.001):

        self.time_dependency_step_size = time_depency_step_size
        # create bounds (upper and lower) for species

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

    def run_iteration(self):
        pass


class SSA():

    def __init__(self, states, transitions, properties, actions,
                 object_removal=None, name="", device="GPU"):
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

        #os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

        # create tensors object to reference the correct tensor class
        # depending on the device
        if (device == "GPU") & (torch.cuda.device_count() > 0):
            self.device = device
            self.tensors = torch.cuda
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
        else:
            self.device = "CPU"
            self.tensors = torch
            torch.set_default_tensor_type(torch.FloatTensor)

        # since state 0 is reserved for no state (no object), start
        for state_nb, state in enumerate(self.states):
            state.number = state_nb + 1
            state.name += "_" + str(state.number)

        for transition in self.transitions:
            if transition.name == "":
                continue
            transition.name = ("transition_from" +str(self.start_state.number) +
                               "_to" + str(self.end_state.number))

    def start(self, nb_simulations, min_time, data_extraction,data_folder,
              max_number_objects=None,
              ignore_errors=False, print_update_time_step=1,
               nb_objects_added_per_step=5,
               dynamically_increase_nb_objects=True, save_states=False,
              use_assertion_checks=True):
        """

        Args:
            nb_simulations (int): Number of simulations to run per parameter
                combination
            min_time (float): minimum time
            data_extraction (DataExtraction object):
            data_folder (string): Folder in which data should be saved
            max_number_objects (int): maximum number of objects allowed to be
                simulated. Determines array size
        Returns:

        """
        # turn off autograd function to reduce memory overhead of pytorch
        # and reduce backend processes
        with torch.no_grad():
            self._start(nb_simulations, min_time, data_extraction,data_folder,
                        max_number_objects, ignore_errors,
                        print_update_time_step,
                        nb_objects_added_per_step,
                        dynamically_increase_nb_objects,save_states,
                        use_assertion_checks)

    def save(self, time_resolution, max_time):
        analysis = analyzer.Analyzer(simulation=self)
        analysis.start(time_resolution, max_time,
                       use_assertion_checks=self.use_assertion_checks)

    def _start(self, nb_simulations, min_time, data_extraction,data_folder,
               max_number_objects=None, ignore_errors=False,
               print_update_time_step=1, nb_objects_added_per_step=5,
               dynamically_increase_nb_objects=True, save_states=False,
               use_assertion_checks=True):
        """

        Args:
            nb_simulations (int): Number of simulations to run per parameter
                combination
            min_time (float): minimum time
            max_number_objects (int): maximum number of objects allowed to be
                simulated. Determines array size
        Returns:
        """
        self.min_time = min_time
        self.data_extraction = data_extraction
        self.data_folder = data_folder
        self.ignore_errors = ignore_errors
        self.nb_objects_added_per_step = nb_objects_added_per_step
        self.dynamically_increase_nb_objects = dynamically_increase_nb_objects
        self.save_states = save_states
        self.use_assertion_checks = use_assertion_checks

        self.all_data = []
        self.all_times = []
        self.all_states = []
        self.all_iteration_nbs = []

        self.initial_memory_used = 0
        self.iteration_memory_used = 0

        self.all_removed_positions = pd.DataFrame()

        # create list with all transitions and then with all actions
        self._all_simulation_parameters = [*self.states,
                                          *self.transitions, *self.actions]

        # create list of length of all model parameters
        simulation_parameter_lengths = [len(parameters.values)
                                        for parameters
                                        in self._all_simulation_parameters]

        if dynamically_increase_nb_objects:
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

        self._zero_tensor = self.tensors.HalfTensor([0])

        self._initialize_parameter_arrays(self._all_simulation_parameters,
                                          simulation_parameter_lengths)

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

        data = self.data_extraction.extract(self)

        self._save_data(data, 0)
        self._save_simulation_parameters()

        # continue simulation until all simulations have reached at least the
        # minimum time
        times_tracked = set()
        iteration_nb = 1
        while True:
            current_min_time = torch.min(self.times)
            if current_min_time >= min_time:
                break
            # print regular current min time in all simulations
            whole_time = current_min_time.item() // print_update_time_step
            if whole_time not in times_tracked:
                print("\n",iteration_nb, "; Current time: ", whole_time)
                times_tracked.add(whole_time)
            self._run_iteration(iteration_nb)
            iteration_nb += 1

        removed_pos_file = os.path.join(self.data_folder,
                                        "removed_param_values.feather")
        self.all_removed_positions.to_feather(removed_pos_file)


    def _run_iteration(self, iteration_nb):
        # create tensor for x (position in neurite), l (length of microtubule)
        # and time
        total_rates = self._get_total_and_single_rates_for_state_transitions()
        reaction_times = self._get_times_of_next_transition(total_rates)

        self._determine_next_transition(total_rates)

        self._determine_positions_of_transitions()

        self._execute_actions_on_objects(reaction_times)

        self._update_object_states()

        # remove objects based on properties
        objects_to_remove = self.object_removal.get_objects_to_remove()
        for object_property in self.properties:
            object_property.array[objects_to_remove] = float("nan")

        self.object_states[objects_to_remove] = 0
        objects_to_remove = None

        self.times += reaction_times

        data = self.data_extraction.extract(self)
        self._save_data(data, iteration_nb)
        del data

        # self.get_tensor_memory()
        # check whether there is a simulation in which all object positions
        # are occupied, if so, increase tensor size to make space for more
        # objects
        if self.dynamically_increase_nb_objects:
            self._add_objects_to_full_tensor()

        # only reduce array size, if not all simulations are done
        current_min_time = torch.min(self.times)
        if current_min_time >= self.min_time:
            return

        self._remove_finished_simulations(iteration_nb)


    def _add_objects_to_full_tensor(self):
        """
        Check whether there is a simulation in which all object positions
        are occupied. If there is, add room for more objects by increasing
        the array size for all arrays with object information.

        Returns: None

        """
        positions_object = torch.max(self.object_states[-1])
        if positions_object == 0:
            return None
        #max_pos_with_object = positions_object[:,0].max()
        #if (max_pos_with_object < (self.object_states.shape[0] - 1)):
        #    return None

        self._simulation_array_size[0] += self.nb_objects_added_per_step
        self.max_number_objects += self.nb_objects_added_per_step
        # if that is the case, increase size of all arrays including object
        # information
        zero_array_float_to_add = torch.zeros((self.nb_objects_added_per_step,
                                               *self.object_states.shape[1:]),
                                              dtype=torch.bfloat16)
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
            max_number_objects_with_state += torch.max(state.initial_condition)

        return max_number_objects_with_state.item()

    def _initialize_parameter_arrays(self, all_simulation_parameters,
                                     simulation_parameter_lengths):
        # go through all model parameters and expand array to simulation
        # specific size
        self.dimension_to_parameter_map = {}
        self.parameter_to_dimension_map = {}
        all_dimensions = [dim for dim
                          in range(len(simulation_parameter_lengths) + 2)]
        for dimension, model_parameters in enumerate(all_simulation_parameters):
            array_dimension = dimension + 2
            expand_dimensions = copy.copy(all_dimensions)
            expand_dimensions.remove(array_dimension)
            # expand dimensions of parameter values to simulation array
            array = np.expand_dims(model_parameters.values, expand_dimensions)
            # save array in object, therefore also change objects saved in
            # self.transitions and self.actions
            array = self.tensors.HalfTensor(array)
            array = array.expand(1,*self._simulation_array_size[1:])
            model_parameters.value_array = array
            # assign model parameter to the correct dimension
            # in the simulation arrays
            self.dimension_to_parameter_map[array_dimension] = model_parameters
            self.parameter_to_dimension_map[model_parameters.name] = dimension
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
            if self.device == "GPU":
                initial_cond = state.initial_condition.cuda()
            else:
                initial_cond = state.initial_condition
            expanded_array = self.tensors.ShortTensor(initial_cond)
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
            if self.device == "GPU":
                initial_cond = state.initial_condition.cuda()
            else:
                initial_cond = state.initial_condition
            expanded_array = self.tensors.ShortTensor(initial_cond)
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
            object_property.array = torch.zeros(self._simulation_array_size,
                                                dtype=torch.bfloat16)
            object_property.array[:] = float("nan")
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

            object_property.array = object_property.array.masked_scatter(object_state_mask, 
                                                                        property_values)
            
            #object_property.array[object_state_mask] = property_values

    def _get_total_and_single_rates_for_state_transitions(self):
        # get number of objects in each state
        nb_objects_all_states = self.tensors.ShortTensor()
        # add 1 to number of states since 0 is not explicitly defined
        for state in range(1,len(self.states) + 1):
            nb_objects = torch.sum(self.object_states == state, dim=0)
            # add a new dimension in first position
            nb_objects = nb_objects[None]
            nb_objects_all_states = torch.cat((nb_objects_all_states,
                                               nb_objects))
        # get rates for all state transitions, depending on number of objects
        # in corresponding start state of transition
        all_transition_rates = self.tensors.HalfTensor()
        for transition in self.transitions:
            if transition.start_state is None:
                # for state 0, the number of objects in state 0 is of course
                # not important
                transition.current_rates = transition.value_array
            else:
                start_state = transition.start_state.number - 1
                transition.current_rates = (transition.value_array *
                                            nb_objects_all_states[start_state])
            # if a time-dependent function is defined, modify the rates by
            # this time-dependent function
            if transition.time_dependency is not None:
                transition.current_rates = (transition.current_rates *
                                            transition.time_dependency(self.times))
            current_rates = transition.current_rates
            all_transition_rates = torch.cat((all_transition_rates,
                                             current_rates.unsqueeze(0)))

        # add current nucleation rate to catastrophe rate for each simulation
        total_rates = torch.sum(all_transition_rates, dim=0)

        return total_rates

    def _get_times_of_next_transition(self, total_rates):
        # get time of next event for each simulation
        exponential_func = torch.distributions.exponential.Exponential
        reaction_times = exponential_func(total_rates,
                                          validate_args=False).sample()
        # print("rate:", total_rates.min(), total_rates.mean(),
        #       "\nreac:", reaction_times.max(),reaction_times.mean(),
        #       reaction_times.min())
        return reaction_times

    def _determine_next_transition(self, total_rates):
        # get which event happened in each simulation
        random_numbers = torch.rand(total_rates.shape, dtype=torch.half)

        # set random number in zero rate positions to >1 to make threshold
        # higher than total rate, thereby preventing any reaction from
        # being executed
        random_numbers[total_rates == 0] = 1.1
        thresholds = total_rates * random_numbers

        # go through each transition and check whether it will occur
        rate_array_shape = self.transitions[0].current_rates.shape
        current_rate_sum = torch.zeros(rate_array_shape, dtype=torch.float)
        all_transitions_mask = torch.zeros(rate_array_shape, dtype=torch.bool)
        for transition in self.transitions:
            current_rate_sum += transition.current_rates
            transition_mask = ((current_rate_sum - thresholds) >=
                               self._zero_tensor)
            # exclude positions for previous transitions from an additional
            # transition to happen
            transition_mask[all_transitions_mask] = False
            # include current transition in mask of all transitions so far
            all_transitions_mask = all_transitions_mask | transition_mask
            transition.simulation_mask = transition_mask

        if self.use_assertion_checks:
            # test whether the expected number of transitions
            # (one per simulation) is observed
            nb_no_transitions = len(torch.nonzero(total_rates == 0))
            expected_total_nb_transitions = np.prod(rate_array_shape[1:])
            nb_transitions = len(torch.nonzero(all_transitions_mask))
            assert expected_total_nb_transitions == (nb_no_transitions +
                                                     nb_transitions)

        return None

    def _determine_positions_of_transitions(self):

        # the transitions masks only tell which reaction happens in each
        # stimulation, but not which object in this simulation is affected
        # To get one random object of the possible objects,
        # first, create mask of index positions, so that each object for each
        # simulation has a unique identifier (index) within this simulation
        # setting all positions where no catastrophe can take place to 0
        for transition in self.transitions:
            transition_mask = transition.simulation_mask
            array_size = self._simulation_array_size
            possible_transition_positions = self.index_array.expand(array_size)
            possible_transition_positions =possible_transition_positions.clone()
            # exclude simulations where the transition did not happen

            no_transition_nb = self._simulation_array_size[0] + 2

            possible_transition_positions[~transition_mask.expand(
                *self.object_states.shape)] = no_transition_nb
            # exclude positions in simulations that were not in the start state
            if transition.start_state is None:
                start_state = 0
            else:
                start_state = transition.start_state.number
            start_state_positions = self.object_states == start_state
            possible_transition_positions[~start_state_positions] =no_transition_nb
            idx_positions = torch.amin(possible_transition_positions,
                                       dim=0, keepdim=True)

            transition_positions = ((possible_transition_positions ==
                                     idx_positions) &
                                    (possible_transition_positions <
                                     no_transition_nb))

            transition.transition_positions = transition_positions
        return None

    def _execute_actions_on_objects(self, reaction_times):
        # execute actions on objects depending on state, before changing state
        for action in self.actions:
            # get a mask that includes all objects on which the action should be
            # executed
            if action.states is None:
                action_positions = self.object_states > 0
            else:
                action_positions = torch.zeros(self.object_states.shape,
                                               dtype=torch.bool)
                for state in action.states:
                    action_positions = (action_positions |
                                        (self.object_states ==
                                         state.number))
            if torch.count_nonzero(action_positions) == 0:
                continue
            object_property_array = action.object_property.array
            sim_array_shape = self.object_states.shape
            action_reaction_times = reaction_times.expand(*sim_array_shape)
            value_array = action.value_array
            value_array = value_array.expand(*self.object_states.shape)

            transformed_property_array = action.operation(object_property_array,
                                                          action_reaction_times,
                                                          value_array,
                                                          action_positions)
            transformed_property_array = transformed_property_array.bfloat16()
            object_property_array = transformed_property_array

            # prevent object properties going above min or max value
            min_property_value = action.object_property.min_value
            if min_property_value is not None:
                objects_below_min = (object_property_array <
                                     min_property_value)
                diff = min_property_value - object_property_array
                object_property_array += diff * objects_below_min

            max_property_value = action.object_property.max_value
            if max_property_value is not None:
                objects_above_max = (object_property_array >
                                     max_property_value)
                diff = object_property_array - max_property_value
                object_property_array -= diff*objects_above_max

        return None

    def _update_object_states(self):
        # update the simulations according to executed transitions
        for transition in self.transitions:
            transition_positions = transition.transition_positions
            if transition.start_state is None:
                start_state = 0
            else:
                start_state = transition.start_state.number
            if transition.end_state is None:
                end_state = 0
            else:
                end_state = transition.end_state.number
            self.object_states[transition_positions] = end_state
            # if state ended in state 0, set property array at position to NaN
            if end_state == 0:
                for object_property in self.properties:
                    object_property.array[transition_positions] = float("nan")
                transition.transition_positions = None
                continue
            # if state started in state 0, add new entry in property array
            if start_state != 0:
                transition.transition_positions = None
                continue
            nb_creations = len(torch.nonzero(transition_positions))
            transition.transition_positions = None
            for object_property in self.properties:
                if type(object_property.start_value) == list:
                    get_property_vals = self._get_random_poperty_values
                    min_value = object_property.start_value[0]
                    max_value = object_property.start_value[1]
                    property_values = get_property_vals(min_value, max_value,
                                                        nb_creations)
                else:
                    property_values = object_property.start_value
                    
                object_property.array = object_property.array.masked_scatter(transition_positions, property_values)
                #object_property.array[transition_positions] = property_values
        return None

    def _get_random_poperty_values(self, min_value,
                                   max_value, nb_objects):
        # scale random number from min_value to max_value
        property_values = (torch.rand((nb_objects), dtype=torch.bfloat16) *
                           (max_value - min_value) + min_value)
        return property_values

    def _remove_finished_simulations(self, iteration_nb):
        # reduce array size every defined number of iterations
        # by checking whether all simulations for a parameter value are done
        dim_list = list(range(len(self.times.shape)))

        all_finished_sim_positions = self._get_finished_simulation(dim_list,
                                                                   iteration_nb)

        self._remove_finished_positions_from_all_arrays(all_finished_sim_positions)

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
        for dim, dim_list in enumerate(all_dim_list[1:]):
            finished_sim_positions = all_finished_sim_positions[dim]
            dim += 1
            for finished_sim_position in finished_sim_positions:
                dim_list.remove(finished_sim_position)
            # make all arrays smaller
            self.times = torch.index_select(self.times, dim=dim,
                                            index=self.tensors.IntTensor(dim_list))
            for property in self.properties:
                property.array = torch.index_select(property.array,
                                                    dim=dim,
                                                    index=
                                                    self.tensors.IntTensor(dim_list))
            self.object_states = torch.index_select(self.object_states,
                                                    dim=dim,
                                                    index=self.tensors.IntTensor(dim_list))

            for sim_parameter in self._all_simulation_parameters:
                sim_parameter.value_array = torch.index_select(sim_parameter.value_array,
                                                             dim=dim,
                                                             index=
                                                             self.tensors.IntTensor(dim_list))

        self._simulation_array_size = [self._simulation_array_size[0],
                                       *self.times.shape[1:]]

        return None

    def _save_data(self, data, iteration_nb):

        # check how much space is free on the GPU (if executed on GPU)
        if self.device == "GPU":
            self.get_tensor_memory()
            torch.cuda.empty_cache()
            total_memory = torch.cuda.get_device_properties(0).total_memory
            reserved_memory = torch.cuda.memory_reserved(0)
            free_memory = (total_memory - reserved_memory)

            if iteration_nb == 0:
                self.initial_memory_used = reserved_memory
            if iteration_nb == 1:
                self.iteration_memory_used = (reserved_memory -
                                              self.initial_memory_used)

            # memory used by data
            total_data_memory = 0
            for tensor in [data, self.times, self.object_states]:
                if type(tensor) == dict:
                    for data_elements in tensor.values():
                        size = (data_elements.element_size() *
                                data_elements.nelement())
                        total_data_memory += size

                else:
                    size = (tensor.element_size() *
                            tensor.nelement())
                    total_data_memory += size


            memory_used = min(self.iteration_memory_used,
                              self.initial_memory_used)
                              
            print(free_memory/1024/1024/1024, 
            memory_used/1024/1024/1024, total_data_memory/1024/1024/1024,
            torch.cuda.memory_allocated(0)/1024/1024/1024,
            torch.cuda.memory_reserved(0)/1024/1024/1024 )
            # if 2x space free on GPU than would be needed for data, keep on GPU
            if free_memory > (2 * (memory_used + total_data_memory)):
                self.all_iteration_nbs.append(iteration_nb)
                self.all_data.append(copy.deepcopy(data))
                self.all_times.append(torch.clone(self.times))
                if self.save_states:
                    self.all_states.append(torch.clone(self.object_states))
                return None

            self.all_iteration_nbs.append(iteration_nb)
            self.all_data.append(data)
            self.all_times.append(self.times)
            if self.save_states:
                self.all_states.append(self.object_states)

            # if not enough space free, move to CPU memory
            all_data_cpu = []
            all_times_cpu = []
            all_object_states_cpu = []
            for data_dict in self.all_data:
                new_data_dict = {}
                for keyword, data in data_dict.items():
                    new_data_dict[keyword] = data.cpu()
                all_data_cpu.append(new_data_dict)
            for times in self.all_times:
                all_times_cpu.append(times.cpu())
            for states in self.all_states:
                all_object_states_cpu.append(states.cpu())
            all_iteration_nbs = copy.copy(self.all_iteration_nbs)

            self.all_iteration_nbs = []
            self.all_data = []
            self.all_times = []
            self.all_states = []

        else:
            all_iteration_nbs = [iteration_nb]
            all_data_cpu = [data]
            all_times_cpu = [self.times]
            all_object_states_cpu = [self.object_states]

        for (iteration_nb,data_cpu,
             times_cpu, object_states_cpu) \
                in zip(all_iteration_nbs, all_data_cpu,
                       all_times_cpu, all_object_states_cpu):

            # save data to hard drive
            for file_name, data_array in data_cpu.items():
                file_path = os.path.join(self.data_folder,
                                         file_name+"_"+str(iteration_nb)+".pt")
                torch.save(data_array, file_path)

            file_path = os.path.join(self.data_folder,
                                     "times_" + str(iteration_nb) + ".pt")
            torch.save(self.times, file_path)

            if self.save_states:
                file_path = os.path.join(self.data_folder,
                                         "states_" + str(iteration_nb) + ".pt")
                torch.save(self.object_states, file_path)

        return None

    def _save_simulation_parameters(self):

        for parameter in self._all_simulation_parameters:
            file_name = "param_"+parameter.name+".pt"
            torch.save(parameter.value_array, os.path.join(self.data_folder,
                                                           file_name))
        return None
