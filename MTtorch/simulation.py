# -*- coding: utf-8 -*-
"""
Write stochastic model of microtubules in neurite
"""

import torch
import copy
import time

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

class simulation():

    def __init__(self, states, transitions, properties, actions, name=""):
        """

        Args:
            states (list of state objects): The state for not present does not
                need to be defined
            transitions (list of state transition objects):
            properties (list of property objects):
            actions (list of action objects):
            name (str): Name of simulation, used for data export and readibility
        """
        self.states = states
        self.transitions = transitions
        self.properties = properties
        self.actions = actions
        self.name = name

        # since state 0 is reserved for no state (no object), start
        for state_nb, state in enumerate(self.states):
            state.number = state_nb + 1

        self.action_funcs = {}
        self.action_funcs["add"] = self._add_to_property
        self.action_funcs["remove"] = self._remove_from_property

    def start(self, nb_simulations, min_time, max_number_objects):
        """

        Args:
            nb_simulations (int): Number of simulations to run per parameter
                combination
            min_time (float): minimum time
            max_number_objects (int): maximum number of objects allowed to be
                simulated. Determines array size

        Returns:

        """

        self.max_number_objects = max_number_objects

        # create list with all transitions and ten with all actions
        all_model_parameters = [*self.transitions, *self.actions]

        # create list of length of all model parameters
        model_parameter_lengths = [len(parameters.values)
                                   for parameters in all_model_parameters]

        # array size contains for each combination of parameters to explore
        self._simulation_array_size = (self.max_number_objects, nb_simulations,
                                       *model_parameter_lengths)

        self._zero_tensor = torch.FloatTensor([0])
        self._one_tensor = torch.FloatTensor([1])

        # go through all model parameters and expand array to simulation
        # specific size
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

        self.times = torch.zeros(self._simulation_array_size)[0, :]

        objects_with_states = self._initialize_object_states()

        self._initialize_object_properties(objects_with_states)

        for action in self.actions:
            array = action.values.expand(self._simulation_array_size)
            action.value_array = array

        # continue simulation until all simulations have reached at least the
        # minimum time
        while torch.min(self.times) < min_time:
            self._run_iteration()

    def _initialize_object_states(self):
        # initial object states, also using defined initial condition of
        # number of objects starting in each state
        self.object_states = torch.zeros(self._simulation_array_size)
        # keep track which objects already have a state set
        # to more easily assign states to the correct positions
        objects_with_states = 0
        for state in self.states:
            if state.initial_condition is None:
                continue
            nb_objects_in_state = state.initial_condition
            start = objects_with_states
            end = objects_with_states + nb_objects_in_state
            if end > self.object_states.shape[0]:
                raise ValueError(f"Initial conditions for states "
                                 f"implied more objects with a state "
                                 f"than the defined maximum number of "
                                 f"objects. After state {state.name} the "
                                 f"total number of objects would be {end} "
                                 f"which is more than the maximum allowed "
                                 f"number {self.max_number_objects}.")
            self.object_states[start:end] = state.number
            objects_with_states += nb_objects_in_state
        return objects_with_states

    def _initialize_object_properties(self, objects_with_states):
        # create tensor for each object property
        # so that each object can have a value for each property
        # also respect initial condition, if defined
        object_state_mask = self.object_states > 0
        for object_property in self.properties:
            object_property.array = torch.zeros(self._simulation_array_size)
            object_property.array = float("nan")
            initial_cond = object_property.initial_condition
            if ((initial_cond is not None) &
                    (type(initial_cond) == type(self.__init__))):
                # if initial condition is defined and a function,
                # get values from function
                property_values = initial_cond(objects_with_states)
            elif ((initial_cond is not None) &
                  (type(initial_cond) == str)):
                # if initial condition is a string == "random
                # then random numbers from min to max val should be generated
                if initial_cond == "random":
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

            elif (object_property.start_value is not None):
                # if no initial cond is defined, but a start value
                # use the start value instead
                property_values = object_property.start_value

            else:
                # if no initial values are defined, create random values between
                # min_value and max_value
                random_property_vals = True

            if random_property_vals:
                get_property_vals = self._get_random_poperty_values
                property_values = get_property_vals(object_property,
                                                    objects_with_states)

            object_property[object_state_mask] = property_values


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
        return None

    def _determine_positions_of_transitions(self):
        # create array of index numbers of same shape than MTs
        index_array = torch.linspace(1, self.max_number_objects+1,
                                     self.max_number_objects+1).view(-1, 1, 1, 1, 1)
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
        return None

    def _execute_actions_on_objects(self, reaction_times):
        # execute actions on objects depending on state, before changing state
        for action in self.actions:
            # get a mask that includes all objects on which the action should be
            # executed
            if action.states is None:
                action_positions = self.object_states > 0
            else:
                action_positions = torch.zeros(self.object_states.shape)
                for state in action.states:
                    action_positions = (action_positions |
                                        (self.object_states ==
                                         state.number))
            property_array = action.object_property.array[action_positions]
            action_reaction_times = reaction_times[action_positions]
            value_array = action.value_array[action_positions]
            if type(action.operation) == str:
                # choose a function of one of the established functions
                error_msg = (f"The action operation {action.operation}"
                             f" is not defined. ")
                error_details = (f"Please either choose one of the "
                                 f"following function names as string: "
                                 f"{', '.join(list(self.action_funcs.keys))}."
                                 f" Or alternatively define a function"
                                 f" for the 'operation' parameter that takes"
                                 f" the property_array, reaction_times and the"
                                 f" value array of the action as parameters.")
                if action.operation not in self.action_funcs:
                    raise ValueError(error_msg + error_details)
                operation_func = self.action_funcs[action.operation]
            # check that the operation is a function object
            elif type(action.operation) == type(self.__init__):
                operation_func = action.operation
            else:
                raise ValueError("Only strings or functions are allowed." +
                                 error_details)

            transformed_property_array = operation_func(property_array,
                                                        action_reaction_times,
                                                        value_array)
            action.object_property.array[action_positions] = transformed_property_array
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
                continue
            # if state started in state 0, add new entry in property array
            if start_state != 0:
                continue
            nb_creations = len(torch.nonzero(transition_positions))
            for object_property in self.properties:
                if object_property.start_value is None:
                    get_property_vals = self._get_random_poperty_values
                    property_values = get_property_vals(object_property,
                                                        nb_creations)
                else:
                    property_values = object_property.start_value
                object_property.array[transition_positions] = property_values
        return None

    def _get_random_poperty_values(self, object_property, nb_objects):
        # scale random number from min_value to max_value
        min_value = object_property.min_value
        max_value = object_property.max_value
        property_values = (torch.rand((nb_objects)) *
                           (max_value - min_value) + min_value)
        return property_values

    def _add_to_property(self, object_property, reaction_times, action_values):
        return object_property + (reaction_times * action_values)

    def _remove_from_property(self, object_property, reaction_times,
                              action_values):
        return object_property - (reaction_times * action_values)



