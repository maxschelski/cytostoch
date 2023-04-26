# -*- coding: utf-8 -*-
"""
Write stochastic model of microtubules in neurite
"""

import torch
import copy
import numpy as np
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

class SSA():

    def __init__(self, states, transitions, properties, actions,
                 object_removal=None, name="",):
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

        # since state 0 is reserved for no state (no object), start
        for state_nb, state in enumerate(self.states):
            state.number = state_nb + 1

        self.action_funcs = {}
        self.action_funcs["add"] = self._add_to_property
        self.action_funcs["remove"] = self._remove_from_property

    def start(self, nb_simulations, min_time, max_number_objects,
              ignore_errors=False, print_update_time_step=1):
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
        self.ignore_errors = ignore_errors

        # create list with all transitions and then with all actions
        all_simulation_parameters = [*self.states,
                                     *self.transitions, *self.actions]

        # create list of length of all model parameters
        simulation_parameter_lengths = [len(parameters.values)
                                        for parameters
                                        in all_simulation_parameters]

        # array size contains for each combination of parameters to explore
        self._simulation_array_size = (self.max_number_objects, nb_simulations,
                                       *simulation_parameter_lengths)

        self._zero_tensor = torch.FloatTensor([0])

        # go through all model parameters and expand array to simulation
        # specific size
        self.dimension_to_parameter_map = {}
        self.parameter_to_dimension_map = {}
        all_dimensions = [dim
                          for dim
                          in range(len(simulation_parameter_lengths) + 2)]
        for dimension, model_parameters in enumerate(all_simulation_parameters):
            array_dimension = dimension + 2
            expand_dimensions = copy.copy(all_dimensions)
            expand_dimensions.remove(array_dimension)
            # expand dimensions of parameter values to simulation array
            array = np.expand_dims(model_parameters.values, expand_dimensions)
            # save array in object, therefore also change objects saved in
            # self.transitions and self.actions
            array = torch.HalfTensor(array)
            array = array.expand(1,*self._simulation_array_size[1:])
            model_parameters.value_array = array
            # assign model parameter to the correct dimension
            # in the simulation arrays
            self.dimension_to_parameter_map[dimension] = model_parameters
            self.parameter_to_dimension_map[model_parameters.name] = dimension

        self.times = torch.zeros(self._simulation_array_size)[0, :]

        # create index array in which each entry has the value of the index of
        # the microtubule in the simulation, thereby multi-D operations
        # on single simulations can be executed
        # create array of index numbers of same shape as whole simulation array
        view_array = [-1] + [1] * (len(self._simulation_array_size) - 1)
        max_number_objects = self.max_number_objects
        self.index_array = torch.linspace(1, max_number_objects,
                                          max_number_objects).view(*view_array)
        self.index_array = self.index_array.expand(self._simulation_array_size)

        self._initialize_object_states()

        self._initialize_object_properties()

        # continue simulation until all simulations have reached at least the
        # minimum time
        times_tracked = set()
        while True:
            current_min_time = torch.min(self.times)
            if current_min_time >= min_time:
                break
            # print regular current min time in all simulations
            whole_time = current_min_time.item() // print_update_time_step
            if whole_time not in times_tracked:
                print("Current time: ", whole_time)
                times_tracked.add(whole_time)
            self._run_iteration()

    def _initialize_object_states(self):
        # initial object states, also using defined initial condition of
        # number of objects starting in each state
        self.object_states = torch.zeros(self._simulation_array_size)

        # keep track which objects already have a state set
        # to assign states to the correct positions
        # due to combinations of different number of objects for different
        # states, keep track of already assigned objects for each state
        object_state_shape = self.object_states.shape
        nb_objects_with_states = torch.zeros((1,*object_state_shape[1:]))
        # first sum all number, to get the total number of object with states
        # (after all states are assigned)
        for state in self.states:
            if state.initial_condition is None:
                continue
            # expand the initial condition array, to add them up
            expanded_array = torch.HalfTensor(state.initial_condition)
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
            expanded_array = torch.HalfTensor(state.initial_condition)
            expanded_array = expanded_array.expand((1,*object_state_shape[1:]))

            object_state_mask = torch.where(self.index_array <=
                                            nb_objects_with_states, True, False)

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
            object_property.array = torch.zeros(self._simulation_array_size)
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

            object_property.array[object_state_mask] = property_values


    def _run_iteration(self):
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

        self.times += reaction_times[0]

    def _get_total_and_single_rates_for_state_transitions(self):
        # get number of objects in each state
        nb_objects_all_states = torch.HalfTensor()
        # add 1 to number of states since 0 is not explicitly defined
        for state in range(len(self.states) + 1):
            nb_objects = torch.sum(self.object_states == state, dim=0)
            # add a new dimension in first position
            nb_objects = nb_objects[None]
            nb_objects_all_states = torch.cat((nb_objects_all_states,
                                               nb_objects))

        # get rates for all state transitions, depending on number of objects
        # in corresponding start state of transition
        all_transition_rates = torch.HalfTensor()
        for transition in self.transitions:
            if transition.start_state is None:
                start_state = 0
            else:
                start_state = transition.start_state.number
            transition.current_rates = (transition.value_array *
                                        nb_objects_all_states[start_state])
            # if a time-dependent function is defined, modify the rates by
            # this time-dependent function
            if transition.time_dependency is None:
                continue
            transition.current_rates *= transition.time_dependency(self.times)
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
        reaction_times = reaction_times.expand(*self.object_states.shape)
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
        current_rate_sum = torch.zeros_like(self.transitions[0].current_rates)
        for transition in self.transitions:
            current_rate_sum += transition.current_rates
            transition_mask = ((current_rate_sum - thresholds) >
                               self._zero_tensor)
            transition_mask = transition_mask.expand(*self.object_states.shape)
            transition.simulation_mask = transition_mask
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
            possible_transition_positions = torch.clone(self.index_array)
            # exclude simulations where the transition did not happen
            possible_transition_positions[~transition_mask] = 0
            # exclude positions in simulations that were not in the start state
            if transition.start_state is None:
                start_state = 0
            else:
                start_state = transition.start_state.number
            start_state_positions = self.object_states == start_state
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
            # if torch.count_nonzero(action_positions) == 0:
            #     continue
            object_property_array = action.object_property.array
            property_array = object_property_array[action_positions]
            action_reaction_times = reaction_times[action_positions]
            value_array = action.value_array
            value_array = value_array.expand(*self.object_states.shape)
            value_array = value_array[action_positions]

            transformed_property_array = action.operation(property_array,
                                                          action_reaction_times,
                                                          value_array)
            object_property_array[action_positions] = transformed_property_array

            # MAKE MAX AND MIN THREHOLDS WORK FOR ARRAY AND INT VALUES!

            # prevent object properties going above min or max value
            min_property_value = action.object_property.min_value
            if min_property_value is not None:
                objects_below_min = (object_property_array <
                                     min_property_value)
                if type(min_property_value) == type(self.object_states):
                    min_property_values = min_property_value[objects_below_min]
                object_property_array[objects_below_min] = min_property_values
            max_property_value = action.object_property.max_value
            if max_property_value is not None:
                objects_above_max = (object_property_array >
                                     max_property_value)
                if type(max_property_value) == type(self.object_states):
                    max_property_value = max_property_value[objects_above_max]
                object_property_array[objects_above_max] = max_property_value
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
                if type(object_property.start_value) == list:
                    get_property_vals = self._get_random_poperty_values
                    min_value = object_property.start_value[0]
                    max_value = object_property.start_value[1]
                    property_values = get_property_vals(min_value, max_value,
                                                        nb_creations)
                else:
                    property_values = object_property.start_value
                object_property.array[transition_positions] = property_values
        return None

    def _get_random_poperty_values(self, min_value,
                                   max_value, nb_objects):
        # scale random number from min_value to max_value
        property_values = (torch.rand((nb_objects)) *
                           (max_value - min_value) + min_value)
        return property_values

    def _add_to_property(self, object_property, reaction_times, action_values):
        return object_property + (reaction_times * action_values)

    def _remove_from_property(self, object_property, reaction_times,
                              action_values):
        return object_property - (reaction_times * action_values)