# -*- coding: utf-8 -*-
"""
Write stochastic model of microtubules in neurite
"""

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
- action stategit pu transition class (includes one transition, rates 
                                 (rate can be supplied as lifetime))
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

        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

        # create tensors object to reference the correct tensor class
        # depending on the device
        if (device.lower() == "gpu") & (torch.cuda.device_count() > 0):
            for GPU_nb in range(torch.cuda.device_count()):
                if torch.cuda.memory_reserved(GPU_nb) == 0:
                    break
            self.device = "cuda:"+str(GPU_nb)
            self.tensors = torch.cuda
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
            torch.set_default_device(self.device)
        else:
            self.device = "cpu"
            self.tensors = torch
            torch.set_default_tensor_type(torch.FloatTensor)

        # since state 0 is reserved for no state (no object), start
        for state_nb, state in enumerate(self.states):
            state.number = state_nb + 1
            state.name += "_" + str(state.number)

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
              max_number_objects=None,
              ignore_errors=False, print_update_time_step=1,
               nb_objects_added_per_step=10,
              max_iters_no_data_extraction=2000,
               dynamically_increase_nb_objects=True, save_states=False,
              use_assertion_checks=True, reset_folder=True):
        """

        Args:
            nb_simulations (int): Number of simulations to run per parameter
                combination
            min_time (float): minimum time
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
                        max_number_objects, ignore_errors,
                        print_update_time_step,
                        nb_objects_added_per_step,
                        max_iters_no_data_extraction,
                        dynamically_increase_nb_objects,save_states,
                        use_assertion_checks, reset_folder)

    def save(self, time_resolution, max_time):
        analysis = analyzer.Analyzer(simulation=self)
        analysis.start(time_resolution, max_time,
                       use_assertion_checks=self.use_assertion_checks)

    def _start(self, nb_simulations, min_time, data_extractions,data_folder,
               max_number_objects=None, ignore_errors=False,
               print_update_time_step=1, nb_objects_added_per_step=10,
                max_iters_no_data_extraction=2000,
               dynamically_increase_nb_objects=True, save_states=False,
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

        self.all_data = []
        self.all_times = []
        self.all_states = []
        self.all_iteration_nbs = []

        self.times_buffer = []
        self.object_states_buffer = []

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

        self._zero_tensor = torch.HalfTensor([0]).to(device=self.device)

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

        # data = self.data_extraction.extract(self)
        #
        # if self.device.find("cuda") != -1:
        #     self._free_gpu_memory()
        #
        # self._save_data(data, 0)

        self._add_data_to_buffer()
        self._save_simulation_parameters()

        # continue simulation until all simulations have reached at least the
        # minimum time
        times_tracked = set()

        iteration_nb = 1
        start_time = time.time()
        self.data_buffer = []
        self.last_data_extraction = 0
        self.last_data_saving = 0

        while True:
            current_min_time = torch.min(self.times)
            if current_min_time >= min_time:
                break
            # print regular current min time in all simulations
            whole_time = current_min_time.item() // print_update_time_step
            if whole_time not in times_tracked:
                print("\n",iteration_nb, "; Current time: ",
                      whole_time * print_update_time_step)
                print(time.time() - start_time)
                start_time = time.time()
                times_tracked.add(whole_time)
                # only print data if the current data was not yet printed
                # and if there is data buffer to be printed
                for keyword, data_extraction in data_extractions.items():
                    if not data_extraction.print_regularly:
                        continue
                    data = data_extraction.extract(self)
                    for data_name, data_array in data.items():
                        mean = torch.nanmean(data_array.to(torch.float))
                        if (not np.isnan(mean.item())) & (mean.item() != 0):
                            print(data_name," : ",mean.item())

            self._run_iteration(iteration_nb)

            iteration_nb += 1

        removed_pos_file = os.path.join(self.data_folder,
                                        "removed_param_values.feather")
        self.all_removed_positions.to_feather(removed_pos_file)

    def _run_iteration(self, iteration_nb):
        # create tensor for x (position in neurite), l (length of microtubule)
        # and time
        start = time.time()
        total_rates = self._get_total_and_single_rates_for_state_transitions()
        reaction_times = self._get_times_of_next_transition(total_rates)
        # print(1, time.time() - start)
        start = time.time()

        # print(self.object_states[:,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        # print(self.properties[0].array[:,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        # print(self.properties[1].array[:,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        # print(self.properties[2].array[:,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        # print(self.properties[1].array[:,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

        # if transition.name == "labile_pausing_to_stable":
        # if len(np.unique(transition_positions[:,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])) == 2:

        # print("State 1:", len(self.object_states[self.object_states == 1]))
        # print("State 2:", len(self.object_states[self.object_states == 2]))

        self._determine_next_transition(total_rates)

        self._determine_positions_of_transitions()
        # print(2, time.time() - start)
        start = time.time()

        self._execute_actions_on_objects(reaction_times)
        # print(3, time.time() - start)
        start = time.time()

        self._update_object_states()
        # print(4, time.time() - start)
        start = time.time()

        # remove objects based on properties
        objects_to_remove = self.object_removal.get_objects_to_remove()
        for object_property in self.properties:
            object_property.array[objects_to_remove] = float("nan")

        self.object_states[objects_to_remove] = 0
        objects_to_remove = None

        self.times += reaction_times

        # print(5, time.time() - start)
        start = time.time()

        # self.get_tensor_memory()
        # check whether there is a simulation in which all object positions
        # are occupied, if so, increase tensor size to make space for more
        # objects
        self._add_data_to_buffer()

        if ((iteration_nb - self.last_data_extraction) >
                self.max_iters_no_data_extraction):
            self.data_buffer.append(self._extract_data())
            self._save_times_and_object_states(iteration_nb)
            self.last_data_extraction = iteration_nb

        if ((iteration_nb - self.last_data_saving) >
                self.max_iters_no_data_extraction):
            all_concat_data = self._concat_data_from_buffer()
            self._save_data(all_concat_data, iteration_nb)
            self.last_data_saving = iteration_nb
            del all_concat_data
            gc.collect

        if self.dynamically_increase_nb_objects:
            positions_object = torch.max(self.object_states[-1])
            if positions_object > 0:
                self._add_objects_to_full_tensor()

                start = time.time()
                if self.last_data_extraction != iteration_nb:
                    self.data_buffer.append(self._extract_data())

                    if self.device.find("cuda") != -1:
                        self._free_gpu_memory()

                    self._save_times_and_object_states(iteration_nb)
                    self.last_data_extraction = iteration_nb

                start = time.time()

        # print(8, time.time() - start)
        start = time.time()
        # only reduce array size, if not all simulations are done
        current_min_time = torch.min(self.times)
        if current_min_time >= self.min_time:
            return

        # reduce array size every defined number of iterations
        # by checking whether all simulations for a parameter value are done
        dim_list = list(range(len(self.times.shape)))

        all_finished_sim_positions = self._get_finished_simulation(dim_list,
                                                                   iteration_nb)
        nb_finished_sim_positions = [len(finished_sim_pos)
                                     for finished_sim_pos
                                     in all_finished_sim_positions]
        if np.max(nb_finished_sim_positions) > 0:
            self._remove_finished_positions_from_all_arrays(all_finished_sim_positions)

            if self.last_data_extraction != iteration_nb:
                self.data_buffer.append(self._extract_data())
                self.last_data_extraction = iteration_nb

                if self.device.find("cuda") != -1:
                    self._free_gpu_memory()

                start = time.time()
                self._save_times_and_object_states(iteration_nb)

            if self.last_data_saving != iteration_nb:
                start = time.time()
                all_concat_data = self._concat_data_from_buffer()
                # print(888, time.time() - start)
                start = time.time()
                self._save_data(all_concat_data, iteration_nb)
                self.last_data_saving = iteration_nb
                # print(999, time.time() - start)
                del all_concat_data
                gc.collect

            # print(11111, time.time() - start)

        # print(9, time.time() - start)
        start = time.time()

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
                                             float("nan"), dtype=torch.float)
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
            array = torch.HalfTensor(array).to(device=self.device)
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
            initial_cond = state.initial_condition
            expanded_array = torch.ShortTensor(initial_cond).to(device=self.device)
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
            expanded_array = torch.ShortTensor(initial_cond).to(device=self.device)
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

                # if there is just one property value (its not a tensor)
                # use another way of assinging values
                # masked_scatter has advantage of much smaller memory footprint
                # but when just assigning one value, memory footprint using
                # mask directly is small
            if type(property_values) != type(object_property.array):
                object_property.array[object_state_mask] = property_values
                continue
            object_property.array = object_property.array.masked_scatter(object_state_mask, 
                                                                        property_values)
        return None

    def _get_total_and_single_rates_for_state_transitions(self):
        # get number of objects in each state
        nb_objects_all_states = torch.ShortTensor().to(device=self.device)
        # add 1 to number of states since 0 is not explicitly defined
        for state in range(1,len(self.states) + 1):
            nb_objects = torch.sum(self.object_states == state, dim=0)
            # add a new dimension in first position
            nb_objects = nb_objects[None]
            nb_objects_all_states = torch.cat((nb_objects_all_states,
                                               nb_objects))
        # get rates for all state transitions, depending on number of objects
        # in corresponding start state of transition
        all_transition_rates = torch.HalfTensor().to(device=self.device)
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
        total_rates = torch.sum(all_transition_rates, dim=0, dtype=torch.float)

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

            # transition_happened = transition_mask[:,0,0,0,0,0,0,0,0,0,
            #                                     0,0,0,0,0,0,0,0,0]
            # if transition_happened:
            #    print(transition.name)

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
            possible_transition_positions = self.index_array.expand(array_size) + 1
            possible_transition_positions =possible_transition_positions.clone()
            # exclude simulations where the transition did not happen

            no_transition_nb = self._simulation_array_size[0] + 3

            possible_transition_positions[~transition_mask.expand(
                *self.object_states.shape)] = no_transition_nb
            # exclude positions in simulations that were not in the start state
            if transition.start_state is None:
                start_state = 0
            else:
                start_state = transition.start_state.number
            start_state_positions = self.object_states == start_state
            possible_transition_positions[~start_state_positions] =no_transition_nb

            # for nucleation of new objects, always start at the leftmost
            # position
            if start_state == 0:
                # get the minimum object index for each simulation
                idx_positions = torch.amin(possible_transition_positions,
                                           dim=0, keepdim=True)
                # since all simulations with the wrong start state and where
                # the transition did not happen at all have object indexes
                # no_transition_nb (which is larger than the largest possible
                # index), the transition positions are the the smallest object
                # indices that are still smaller than no_transition_nb
                transition_positions = ((possible_transition_positions ==
                                         idx_positions) &
                                        (possible_transition_positions <
                                         no_transition_nb))
            # for any other transition, a random object must be selected!
            # Otherwise there will be a strong bias towards the first objects
            # undergoing all dynamics
            else:
                # first set no transition points to 0, since to get the random
                # position the maximum possible position must be used
                # (which must not be the artificial no_transition_nb)
                no_transition_mask = (possible_transition_positions ==
                                      no_transition_nb)
                possible_transition_positions[no_transition_mask] = 0
                # use the maximum allowed transition position for each
                # simulation and multiply it with a random uniform number
                # between 0 and 1
                max_positions = torch.amax(possible_transition_positions,
                                           dim=0, keepdim=True)
                rand_pos = torch.rand((1,
                                       *possible_transition_positions.shape[1:]
                                       ))
                rand_pos = rand_pos * max_positions
                # now get the position closest to this random position by
                # subtracting the random pos from the possible
                # transition_positions
                distance_from_random = possible_transition_positions - rand_pos
                distance_from_random = torch.abs(distance_from_random)
                # then set the points not allowed for transition to the high
                # no_transition_nb
                distance_from_random[no_transition_mask] = no_transition_nb
                # then take the minimum number, which is the minimum distance
                # from the random number
                transition_idxs = torch.amin(distance_from_random,
                                             dim=0, keepdim=True)
                # print(transition_idxs[:,0,0,0,0,0,0,0,0,0,0,0,0,0])
                # dasd
                transition_positions = ((distance_from_random ==
                                         transition_idxs) &
                                        (distance_from_random <
                                         no_transition_nb))

            transition.transition_positions = transition_positions
        return None

    def _execute_actions_on_objects(self, reaction_times):
        # execute actions on objects depending on state, before changing state
        for action in self.actions:
            # get a mask that includes all objects on which the action should be
            # executed
            start = time.time()
            if action.states is None:
                action_positions = self.object_states > 0
                # assume that almost always in all the simulations at least one
                # there is at least one object
                # even if not, the mask multiplication 
                # will prevent wrong values
                any_objects = True
            else:
                action_positions = torch.zeros(self.object_states.shape,
                                               dtype=torch.bool)
                for state in action.states:
                    action_positions = (action_positions |
                                        (self.object_states ==
                                         state.number))
                any_objects = action_positions.any()
            if not any_objects:
                continue
            # print(3.1, time.time() - start)
            start = time.time()
            object_property_array = action.object_property.array
            sim_array_shape = self.object_states.shape
            action_reaction_times = reaction_times.expand(*sim_array_shape)
            value_array = action.value_array
            # print(3.2, time.time() - start)
            start = time.time()
            value_array = value_array.expand(*self.object_states.shape)

            #transformed_property_array = object_property_array + action_reaction_times*value_array*action_positions
            transformed_property_array = action.operation(object_property_array,
                                                          action_reaction_times,
                                                          value_array,
                                                          action_positions)

            # if action.name == "Act_MT_growth":
            #     print(transformed_property_array[:,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
            #     if np.nanmin(transformed_property_array[:,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]) < 0:
            #         print(self.object_states[:,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
            #         print(transformed_property_array[:,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

            # print(3.3, time.time() - start)
            start = time.time()
            # transformed_property_array = transformed_property_array.bfloat16()
            object_property_array = transformed_property_array

            # print(object_property_array[:,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
            # print(3.4, time.time() - start)
            start = time.time()

            # prevent object properties going below min or above max value
            min_property_value = action.object_property.min_value
            if min_property_value is not None:
                objects_below_min = (object_property_array <
                                     min_property_value)
                # if max property is not a tensor, but just a single value
                if type(min_property_value) != type(object_property_array):
                    object_property_array[objects_below_min] = min_property_value
                else:
                    diff = min_property_value - object_property_array
                    object_property_array += diff * objects_below_min

            # print(3.5, time.time() - start)
            # start = time.time()
            max_property_value = action.object_property.max_value
            if max_property_value is not None:
                objects_above_max = (object_property_array >
                                     max_property_value)
                # if max property is not a tensor, but just a single value
                if type(max_property_value) != type(object_property_array):
                    object_property_array[objects_above_max] = max_property_value
                else:
                    diff = object_property_array - max_property_value
                    object_property_array -= diff*objects_above_max
            action.object_property.array = object_property_array
            # print(3.6, time.time() - start)

            start = time.time()
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

            # if transition.name == "labile_pausing_to_stable":
            #     if len(np.unique(transition_positions[:,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])) == 2:
            #         print(self.object_states[:,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
            #         print(self.properties[0].array[:,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
            #         print(self.properties[1].array[:,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
            #         print(self.properties[2].array[:,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
            #         dasd

            # transfer values from one property to another upon transition
            if transition.transfer_property is not None:
                source = transition.transfer_property[0].array
                target = transition.transfer_property[1].array
                target[transition_positions] += source[transition_positions]
                source[transition_positions] = 0

            # set defined property to zero
            if transition.property_set_to_zero is not None:
                transition.property_set_to_zero.array[transition_positions] = 0

            # save all properties defined in saved_properties
            if transition.saved_properties is not None:
                for object_property in transition.saved_properties:
                    if object_property.saved is None:
                        empty_array = torch.zeros_like(object_property.array)
                        object_property.saved = empty_array
                    vals_to_save = object_property.array[transition_positions]
                    object_property.saved[transition_positions] = vals_to_save

            # Retrieve all properties defined in retrieved_properties
            if transition.retrieved_properties is not None:
                for object_property in transition.retrieved_properties:
                    if object_property.saved is None:
                        continue
                    saved_values = object_property.saved[transition_positions]
                    object_property.array[transition_positions] = saved_values
                    #reset saved vaues
                    object_property.saved[transition_positions] = float("nan")

            # if state ended in state 0, set property array at position to NaN
            if end_state == 0:
                for object_property in self.properties:
                    object_property.array[transition_positions] = float("nan")
                transition.transition_positions = None
                continue
            # if state started in state 0, add new entry in property arrays
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

                # if there is just one property value (its not a tensor)
                # use another way of assinging values
                # masked_scatter has advantage of much smaller memory footprint
                # but when just assigning one value, memory footprint using
                # mask directly is small
                if type(property_values) != type(object_property.array):
                    object_property.array[
                        transition_positions] = property_values
                    continue
                object_property.array = object_property.array.masked_scatter(transition_positions,
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
            dim_list_tensor = torch.IntTensor(dim_list).to(device=self.device)
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

        times_array = torch.concat(self.times_buffer)
        file_path = os.path.join(self.data_folder,
                                    "times_" + str(iteration_nb) + ".pt")
        torch.save(times_array, file_path)

        self.times_buffer = []

        if self.save_states:
            object_state_array = torch.concat(self.object_states_buffer)
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
