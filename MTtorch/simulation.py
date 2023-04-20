# -*- coding: utf-8 -*-
"""
Write stochastic model of microtubules in neurite
"""

import torch
import numpy as np
import copy
import time

nb_simulations = 100

start_number_MTs = 0

max_number_MTs = 1000

max_x = 20

max_t = 180

MT_growth_speed = 15

# define parameters to explore
MT_lifetimes = np.linspace(1, 60, 10)
max_nucleation_rates = np.linspace(1, 10, 10)
MTRF_speeds = np.linspace(0, 2, 10)


def nucleation(time_array, min_nucleation_fraction=0.1):
    """
    Get nucleation rate at defined time
    """
    # 1 has to be added to sin function to prevent negative nucleation
    # then normalize nucleation to be within min_nucleation_fraction and 1
    return ((torch.sin(time_array) + 1) / (2 / (1 - min_nucleation_fraction))
            + min_nucleation_fraction)



"""
This is for single type modeling! (e.g. just MTs) 
What to model?
- allow different action states of MTs and different state variables
- define transition rates between MT states
- define deterministic actions for states (grow, Pause, MT-RF);
    each action can be defined for multiple states
- allow different parameter value ranges
What classes needed?
- simulation class (executes the simulation)
- properties class (e.g. length and position)
- action class (defines what happens to a property, gets property supplied and
                what happens with it)

- state class as dictionary of state with actions
- action state transition class (includes one transition, rates 
                                 (rate can be supplied as lifetime))
"""


class simulation():

    def __init__(self, states, transitions, properties, actions):
        """

        Args:
            states (list of state objects):
            transitions (list of state transition objects):
            properties (list of property objects):
            actions (list of action objects):
        """
        self.states = states
        self.transitions = transitions
        self.properties = properties
        self.actions = actions

        self.action_funcs = {}
        self.action_funcs["add"] = self._add_to_property
        self.action_funcs["remove"] = self._remove_from_property

    def start(self, nb_simulations, min_time, max_number_objects):
        """

        Args:
            nb_simulations (int): Number of simulations to run per parameter
                combination
            min_time (float): maximum time
            max_number_objects (int): maximum number of objects allowed to be
                simulated

        Returns:

        """

        # create list with all transitions and ten with all actions
        all_model_parameters = [*self.transitions, *self.actions]

        # create list of length of all model parameters
        model_parameter_lengths = [len(parameters.values)
                                   for parameters in all_model_parameters]

        # array size contains for each combination of parameters to explore
        self._simulation_array_size = (max_number_objects, nb_simulations,
                                       *model_parameter_lengths)

        self._zero_tensor = torch.FloatTensor([0])
        self._one_tensor = torch.FloatTensor([1])

        # go through all model parameters
        self.dimension_to_parameter_map = {}
        self.parameter_to_dimension_map = {}
        all_dimensions = [dim for dim in len(model_parameter_lengths)]
        for dimension, model_parameters in enumerate(all_model_parameters):
            array_dimension = dimension + 2
            expand_dimensions = copy.copy(all_dimensions)
            expand_dimensions.remove(array_dimension)
            # expand dimensions of parameter values to simulation array
            array = model_parameters.values.expand_dims(expand_dimensions)
            # save array in object, therefore also change objects saved in
            # self.transitions and self.actions
            model_parameters.value_array = array
            # assign model parameter to the correct dimension
            # in the simulation arrays
            self.dimension_to_parameter_map[dimension] = model_parameters
            self.parameter_to_dimension_map[model_parameters.name] = dimension

        # create tensor for each object property
        # so that each object can have a value for each property
        for object_property in self.properties:
            object_property.array = torch.zeros(self._simulation_array_size)
            object_property.array = float("nan")

        self.times = torch.zeros(self._simulation_array_size)[0, :]

        # add support for initial condition. But so far all objects start
        # in the same state
        self.object_states = torch.zeros(self._simulation_array_size)

        for action in self.actions:
            array = action.values.expand(self._simulation_array_size)
            action.value_array = array

        while torch.min(self.times) < min_time:
            self._run_iteration()

    def _run_iteration(self):
        # create tensor for x (position in neurite), l (length of microtubule)
        # and time
        total_rates = self._get_total_and_single_rates_for_state_transitions()

        reaction_times = self._get_times_of_next_transition(total_rates)

        self._determine_next_transition(total_rates)

        self._determine_positions_of_transitions()

        self._execute_actions_on_objects(reaction_times)

        self._update_object_states()

        self.times += reaction_times[0]

    def _get_total_and_single_rates_for_state_transitions(self):
        # get number of objects in each state
        nb_objects_all_states = torch.HalfTensor()
        # add 1 to number of states since 0 is not explicitly defined
        for state in range(len(self.states) + 1):
            nb_objects = torch.sum(self.object_states == state, dim=0)
            nb_objects = torch.unsqueeze(nb_objects, dim=(0,1))
            nb_objects_all_states = torch.cat(nb_objects_all_states, nb_objects)

        # get rates for all state transitions, depending on number of objects
        # in corresponding start state of transition
        all_transition_rates = torch.HalfTensor()
        for transition in self.transitions:
            start_state = transition.start_state
            transition.current_rates = (transition.rates *
                                        nb_objects_all_states[start_state])
            # if a time-dependent function is defined, modify the rates by
            # this time-dependent function
            if transition.time_dependency is None:
                continue
            transition.current_rates *= transition.time_dependency(self.times)
            all_transition_rates = torch.cat(all_transition_rates,
                                             transition.current_rates.unsqueeze(0))

        # add current nucleation rate to catastrophe rate for each simulation
        total_rates = torch.sum(all_transition_rates, dim=0)

        return total_rates

    def _get_times_of_next_transition(self, total_rates):
        # get time of next event for each simulation
        exponential_func = torch.distributions.exponential.Exponential
        reaction_times = exponential_func(total_rates,
                                          validate_args=False).sample()
        return reaction_times

    def _determine_next_transition(self, total_rates):
        # get which event happened in each simulation
        random_numbers = torch.rand(total_rates.shape)
        # set random number in zero rate positions to >1 to make threshold
        # higher than total rate, thereby preventing any reaction from
        # being executed
        random_numbers[total_rates == 0] = 1.1
        thresholds = total_rates * random_numbers

        # go through each transition and check whether it will occur
        current_rate_sum = torch.HalfTensor()
        for transition in self.transitions:
            current_rate_sum += transition.current_rates
            transition_mask = ((current_rate_sum - thresholds) >
                               self.zero_tensor)
            transition_mask  = transition_mask.expand(*self.object_states.shape)
            transition.simulations_mask = transition_mask

    def _determine_positions_of_transitions(self):
        # create array of index numbers of same shape than MTs
        max_number_objects = self.object_states.shape[0]
        index_array = torch.linspace(1, max_number_objects+1,
                                     max_number_objects+1).view(-1, 1, 1, 1, 1)
        index_array = index_array.expand(*self.object_states.shape)

        # the transitions masks only tell which reaction happens in each
        # stimulation, but not which object in this simulation is affected
        # To get one random object of the possible objects,
        # first, create mask of index positions, so that each object for each
        # simulation has a unique identifier (index) within this simulation
        # setting all positions where no catastrophe can take place to 0
        for transition_nb, transition in enumerate(self.transitions):
            transition_mask = transition.simulation_mask[transition_nb]
            possible_transition_positions = torch.clone(index_array)
            # exclude simulations where the transition did not happen
            possible_transition_positions[~transition_mask] = 0
            # exclude positions in simulations that were not in the start state
            start_state_positions = self.object_states == transition.start_state
            possible_transition_positions[~start_state_positions] = 0
            idx_positions = torch.amax(possible_transition_positions,
                                       dim=0, keepdim=True)
            transition_positions = torch.where((possible_transition_positions ==
                                                idx_positions) &
                                               (possible_transition_positions >
                                                0), True, False)
            transition.transition_positions = transition_positions

    def _execute_actions_on_objects(self, reaction_times):
        # execute actions on objects depending on state, before changing state
        for action in self.actions:
            action_positions = self.object_states == action.state
            property_array = action.object_property.array
            if type(action.operation) == str:
                # choose function of the established functions
                error_msg = (f"The action operation {action.operation}"
                             f" is not defined. Please choose one of the "
                             f"following function names: "
                             f"{', '.join(list(self.action_funcs.keys))}.")
                if action.operation not in self.action_funcs:
                    raise ValueError(error_msg)
                continue
            transformed_property_array = action.operation(property_array,
                                                          reaction_times,
                                                          action.value_array)
            action.object_property.array = transformed_property_array

    def _update_object_states(self):
        # update the simulations according to executed transitions
        for transition in self.transitions:
            transition_positions = transition.transition_positions
            self.object_states[transition_positions] = transition.end_state
            # if state ended in state 0, set property array at position to NaN
            if transition.end_state == 0:
                for object_property in self.properties:
                    object_property.array[transition_positions] = float("nan")
                continue
            # if state started in state 0, add new entry in property array
            if transition.start_state != 0:
                continue
            nb_creations = len(torch.nonzero(transition_positions))
            for object_property in self.properties:
                if object_property.start_value is None:
                    # scale random number from min_value to max_value
                    min_value = object_property.min_value
                    max_value = object_property.max_value
                    property_values = (torch.rand((nb_creations)) *
                                       (max_value - min_value) + min_value)
                else:
                    property_values = object_property.start_value
                object_property.array[transition_positions] = property_values

    def _add_to_property(self, object_poperty, reaction_times, action_values):
        return object_poperty + (reaction_times * action_values)

    def _remove_from_property(self, object_poperty, reaction_times, action_values):
        return object_poperty - (reaction_times * action_values)

class object_property():

    def __init__(self, name, min_value=0, max_value=1, start_value=None):
        """
        Args:
            name (String): Name of property
            min_value (float): Minimum allowed value
            max_value (float): Maximum allowed value
            start_value (float): Value at which new objects will be initialized
        """
        self.name = name
        self.min_value = min_value
        self.max_value = max_value
        self.start_value = start_value

        # initialize variables used in simulation
        self.value_array = torch.HalfTensor([])


class action():

    def __init__(self, name, state, object_property, operation, values):
        """

        Args:
            name (String): Name of action
            state (state object):
            object_property (object_property object):
            operation (func): function that takes the property tensor,
                a tensor of the values (same shape as the property tensor) and
                the time tensor and then outputs the transformed property tensor
                Alternatively use one of the following standard_positions:
                "add", "subtract",
            values (1D iterable):

        """
        self.name = name
        self.state = state
        self.operation = operation
        self.values = torch.HalfTensor(values)
        self.object_property = object_property

        # define variables which will be used in simulation
        self.value_array = torch.HalfTensor([])


class state():

    def __init__(self, name):
        self.name = name


class state_transition():

    def __init__(self, start_state, end_state, rates=None, lifetimes=None,
                 time_dependency = None):
        """

        Args:
            start_state (String): Name of state at which the transition starts
            end_state (String): Name of state at which the transition ends
            rates (Iterable): 1D Iterable (list, numpy array) of all rates which
                should be used; rates or lifetimes need to be defined
            lifetimes (Iterable): 1D Iterable (list, numpy array) of all
                lifetimes which should be used, will be converted to rates;
                rates or lifetimes need to be defined
            time_dependency (func): Function that takes a torch tensor
                containing the current timepoints as input and converts it to a
                tensor of factors (using only pytorch functions)
                to then be multiplied with the rates. It is recommended to have
                the time dependency function range from 0 to 1, which makes the
                supplied rates maximum rates.
        """
        if (rates is None) & (lifetimes = None):
            return ValueError("Rates or lifetimes need to be specified for "
                              "the state transition from state"
                              f"{start_state} to state {end_state}.")
        self.start_state = start_state
        self.end_state = end_state
        lifetime_to_rates_factor = torch.log(torch.FloatTensor([2]))
        if rates is None:
            self.rates = lifetime_to_rates_factor / lifetimes
        else:
            self.rates = rates
        self.values = torch.HalfTensor(rates)
        self.time_dependency = time_dependency

        # initialize variable that will be filled during simulation
        self.value_array = torch.HalfTensor([])
        self.simulation_mask = torch.HalfTensor([])
        self.transition_positions = torch.HalfTensor([])
