import numpy as np

# NUMBA_DEBUG=1
# NUMBA_DEVELOPER_MODE = 1
# NUMBA_DEBUGINFO = 1
import numba
import numba.cuda.random
import math
from numba import cuda


def _execute_simulation_gpu(object_states, properties_array, times,
                        nb_simulations, nb_parameter_combinations,
                            parameter_value_array,
                        transition_parameters, all_transition_states,
                        action_parameters, action_state_array,
                        all_action_properties, action_operation_array,
                        current_transition_rates,
                        property_start_values,
                        property_min_values, property_max_values,
                        all_transition_tranferred_vals,
                        all_transition_set_to_zero_properties,
                        all_object_removal_properties,
                        object_removal_operations,

                        nb_objects_all_states,
                        total_rates, reaction_times,
                        current_transitions,
                        all_transition_positions,

                        current_timepoint_array, timepoint_array,
                        object_state_time_array, properties_time_array,
                        time_resolution, min_time, save_initial_state,
                            _, rng_states, simulation_factor, parameter_factor
                        ):
    # np.random.seed(seed)
    iteration_nb = 0

    nb_processes = cuda.gridsize(1)
    thread_id = cuda.grid(1)
    grid = cuda.cg.this_grid()

    total_nb_simulations = nb_simulations * nb_parameter_combinations
    # print(cuda.gridsize(1), total_nb_simulations)
    current_sim_nb = thread_id
    while current_sim_nb < total_nb_simulations:
        # For each parameter combination the defined number of simulations are done
        param_id = int(math.floor(current_sim_nb / nb_simulations))
        sim_id = int(current_sim_nb - param_id * nb_simulations)
        while True:
            _run_iteration(sim_id, param_id, object_states, properties_array,
                           times, parameter_value_array,
                           transition_parameters, all_transition_states,
                           action_parameters, action_state_array,
                           all_action_properties, action_operation_array,
                           current_transition_rates,
                           property_start_values,
                           property_min_values, property_max_values,
                           all_transition_tranferred_vals,
                           all_transition_set_to_zero_properties,
                           all_object_removal_properties,
                           object_removal_operations,

                           nb_objects_all_states,
                           total_rates, reaction_times,
                           current_transitions,
                           all_transition_positions,

                           current_timepoint_array, timepoint_array,
                           object_state_time_array, properties_time_array,
                           time_resolution, save_initial_state,
                           rng_states, simulation_factor, parameter_factor
                           )
            if (current_timepoint_array[sim_id, param_id] >=
                    math.floor(min_time/time_resolution[0])):
                break
            iteration_nb += 1
        current_sim_nb += nb_processes
    grid.sync()
    numba.cuda.syncthreads()

def _execute_simulation_cpu(object_states, properties_array, times,
                        nb_simulations, nb_parameter_combinations,
                            parameter_value_array,
                            transition_parameters, all_transition_states,
                            action_parameters, action_state_array,
                        all_action_properties,action_operation_array,
                        current_transition_rates,
                        property_start_values,
                        property_min_values, property_max_values,
                        all_transition_tranferred_vals,
                        all_transition_set_to_zero_properties,
                        all_object_removal_properties,
                        object_removal_operations,

                        nb_objects_all_states,
                        total_rates, reaction_times,
                        current_transitions,
                        all_transition_positions,

                        current_timepoint_array, timepoint_array,
                        object_state_time_array, properties_time_array,
                        time_resolution, min_time, save_initial_state, seed,
                        rng_states = None
                        ):
    np.random.seed(seed)
    iteration_nb = 0

    thread_id = 0

    # For each parameter combination the defined number of simulations are done
    param_id = int(math.floor(thread_id / nb_simulations))
    sim_id = int(thread_id - param_id * nb_simulations)

    if thread_id < (nb_simulations * nb_parameter_combinations):
        while True:
            _run_iteration(sim_id, param_id, object_states, properties_array, times,
                           parameter_value_array,
                           transition_parameters, all_transition_states,
                           action_parameters, action_state_array,
                           all_action_properties, action_operation_array,
                           current_transition_rates,
                           property_start_values,
                           property_min_values, property_max_values,
                           all_transition_tranferred_vals,
                           all_transition_set_to_zero_properties,
                           all_object_removal_properties,
                           object_removal_operations,

                           nb_objects_all_states,
                           total_rates, reaction_times,
                           current_transitions,
                           all_transition_positions,

                           current_timepoint_array, timepoint_array,
                           object_state_time_array, properties_time_array,
                           time_resolution, save_initial_state, rng_states
                           )
            if current_timepoint_array[sim_id, param_id] >= math.floor(min_time/
                                                               time_resolution[0]):
                break
            iteration_nb += 1
            # print(properties_array[0][object_states == 3])
            # print(properties_array[1][object_states == 3])
            # print(properties_array[2][object_states == 3])
            # print(properties_array[1][object_states == 1] +
            #       properties_array[0][object_states == 1])


def _run_iteration(sim_id, param_id, object_states, properties_array, times,
                   parameter_value_array,
                   transition_parameters, all_transition_states,
                   action_parameters, action_state_array,
                   all_action_properties, action_operation_array,
                   current_transition_rates,
                   property_start_values,
                   property_min_values, property_max_values,
                   all_transition_tranferred_vals,
                   all_transition_set_to_zero_properties,
                   all_object_removal_properties,
                   object_removal_operations,

                   nb_objects_all_states,
                   total_rates, reaction_times,
                   current_transitions,
                   all_transition_positions,

                   current_timepoint_array, timepoint_array,
                   object_state_time_array, properties_time_array,
                   time_resolution, save_initial_state, rng_states,
                   simulation_factor=None, parameter_factor=None
                   ):

    # create tensor for x (position in neurite), l (length of microtubule)
    # and time
    # start = time.time()
    # UPDATE nb_objects_all_states in each iteration at each iteration
    # thereby there is no need to go through all object states
    # get_rates_func = _get_total_and_single_rates_for_state_transitions

    # increase performance through an array for each object state
    # of which transitions are influenced by it
    # then don't recalculate the transition rates each iteration
    # but just update the few rates affected by the changed object state
    _get_rates = _get_total_and_single_rates_for_state_transitions
    _get_rates(parameter_value_array, transition_parameters,
               all_transition_states, current_transition_rates, total_rates,
               nb_objects_all_states, sim_id, param_id)

    get_reaction_times = _get_times_of_next_transition
    get_reaction_times(total_rates, reaction_times, sim_id, param_id,
                       rng_states, simulation_factor, parameter_factor)

    _determine_next_transition(total_rates, current_transition_rates,
                               current_transitions, sim_id, param_id,
                               rng_states, simulation_factor, parameter_factor)

    # speed up searching for xth object with correct state
    # by keeping track of the position of each object in each state
    _determine_positions_of_transitions(current_transitions,
                                        all_transition_states,
                                        nb_objects_all_states,
                                        all_transition_positions,
                                        object_states,
                                        sim_id, param_id, rng_states,
                                        simulation_factor, parameter_factor)

    _execute_actions_on_objects(parameter_value_array, action_parameters,
                                action_state_array,
                                properties_array,
                                property_min_values,
                                property_max_values,
                                all_action_properties,
                                action_operation_array,
                                object_states,
                                reaction_times, sim_id, param_id)

    _update_object_states(current_transitions, all_transition_states,
                          all_transition_positions, object_states,
                          nb_objects_all_states,
                          properties_array,
                          all_transition_tranferred_vals,
                          all_transition_set_to_zero_properties,
                          property_start_values,
                          sim_id, param_id,
                          rng_states, simulation_factor, parameter_factor)

    _remove_objects(all_object_removal_properties, object_removal_operations,
                    object_states, properties_array, sim_id, param_id)

    times[0, sim_id, param_id] = (times[0, sim_id, param_id] +
                                  reaction_times[sim_id, param_id])

    _save_values_with_temporal_resolution(timepoint_array,
                                          object_state_time_array,
                                          properties_time_array,
                                          current_timepoint_array, times,
                                          object_states, properties_array,
                                          time_resolution, save_initial_state,
                                          sim_id, param_id)

def _get_random_number():
    pass

def _decorate_all_functions_for_cpu():
    global _get_random_number
    if not isinstance(_get_random_number,
                      numba.core.registry.CPUDispatcher):
        _get_random_number = numba.njit(_get_random_number_cpu)

    global _execute_simulation_cpu
    if not isinstance(_execute_simulation_cpu,
                      numba.core.registry.CPUDispatcher):
        _execute_simulation_cpu = numba.njit(_execute_simulation_cpu)

    global _run_iteration
    if not isinstance(_run_iteration,
                      numba.core.registry.CPUDispatcher):
        _run_iteration = numba.njit(_run_iteration)

    global _get_total_and_single_rates_for_state_transitions
    if not isinstance(_get_total_and_single_rates_for_state_transitions,
                      numba.core.registry.CPUDispatcher):
        _get_total_and_single_rates_for_state_transitions = numba.njit(
            _get_total_and_single_rates_for_state_transitions)

    global _get_times_of_next_transition
    if not isinstance(_get_times_of_next_transition,
                      numba.core.registry.CPUDispatcher):
        _get_times_of_next_transition = numba.njit(_get_times_of_next_transition)

    global _determine_next_transition
    if not isinstance(_determine_next_transition,
                      numba.core.registry.CPUDispatcher):
        _determine_next_transition = numba.njit(_determine_next_transition)

    global _determine_positions_of_transitions
    if not isinstance(_determine_positions_of_transitions,
                      numba.core.registry.CPUDispatcher):
        _determine_positions_of_transitions = numba.njit(
            _determine_positions_of_transitions)

    global _execute_actions_on_objects
    if not isinstance(_execute_actions_on_objects,
                      numba.core.registry.CPUDispatcher):
        _execute_actions_on_objects = numba.njit(_execute_actions_on_objects)

    global _update_object_states
    if not isinstance(_update_object_states,
                      numba.core.registry.CPUDispatcher):
        _update_object_states = numba.njit(_update_object_states)

    global _save_values_with_temporal_resolution
    if not isinstance(_save_values_with_temporal_resolution,
                      numba.core.registry.CPUDispatcher):
        _save_values_with_temporal_resolution = numba.njit(
            _save_values_with_temporal_resolution)

    global _remove_objects
    if not isinstance(_remove_objects,
                      numba.core.registry.CPUDispatcher):
        _remove_objects = numba.njit(_remove_objects)


def _decorate_all_functions_for_gpu(debug=False):
    opt = debug != True
    global _get_random_number
    if not isinstance(_get_random_number,
                      numba.cuda.dispatcher.CUDADispatcher):
        _get_random_number = numba.cuda.jit(_get_random_number_gpu, debug=debug,
                                            opt=opt)

    global _execute_simulation_gpu
    if not isinstance(_execute_simulation_gpu,
                      numba.cuda.dispatcher.CUDADispatcher):
        _execute_simulation_gpu = numba.cuda.jit(_execute_simulation_gpu,
                                                 debug=debug, opt=opt)

    global _run_iteration
    if not isinstance(_run_iteration,
                      numba.cuda.dispatcher.CUDADispatcher):
        _run_iteration = numba.cuda.jit(_run_iteration, debug=debug, opt=opt)

    global _get_total_and_single_rates_for_state_transitions
    if not isinstance(_get_total_and_single_rates_for_state_transitions,
                      numba.cuda.dispatcher.CUDADispatcher):
        _get_total_and_single_rates_for_state_transitions = numba.cuda.jit(
        _get_total_and_single_rates_for_state_transitions, debug=debug, opt=opt)

    global _get_times_of_next_transition
    if not isinstance(_get_times_of_next_transition,
                      numba.cuda.dispatcher.CUDADispatcher):
        _get_times_of_next_transition = numba.cuda.jit(
            _get_times_of_next_transition, debug=debug, opt=opt)

    global _determine_next_transition
    if not isinstance(_determine_next_transition,
                      numba.cuda.dispatcher.CUDADispatcher):
        _determine_next_transition = numba.cuda.jit(_determine_next_transition,
                                                    debug=debug, opt=opt)

    global _determine_positions_of_transitions
    if not isinstance(_determine_positions_of_transitions,
                      numba.cuda.dispatcher.CUDADispatcher):
        _determine_positions_of_transitions = numba.cuda.jit(
            _determine_positions_of_transitions, debug=debug, opt=opt)

    global _execute_actions_on_objects
    if not isinstance(_execute_actions_on_objects,
                      numba.cuda.dispatcher.CUDADispatcher):
        _execute_actions_on_objects = numba.cuda.jit(
            _execute_actions_on_objects, debug=debug, opt=opt)

    global _update_object_states
    if not isinstance(_update_object_states,
                      numba.cuda.dispatcher.CUDADispatcher):
        _update_object_states = numba.cuda.jit(_update_object_states,
                                               debug=debug, opt=opt)

    global _save_values_with_temporal_resolution
    if not isinstance(_save_values_with_temporal_resolution,
                      numba.cuda.dispatcher.CUDADispatcher):
        _save_values_with_temporal_resolution = numba.cuda.jit(
            _save_values_with_temporal_resolution, debug=debug, opt=opt)

    global _remove_objects
    if not isinstance(_remove_objects,
                      numba.cuda.dispatcher.CUDADispatcher):
        _remove_objects = numba.cuda.jit(_remove_objects, debug=debug, opt=opt)


def _get_total_and_single_rates_for_state_transitions(parameter_value_array,
                                                      transition_parameters,
                                                      all_transition_states,
                                                      current_transition_rates,
                                                      total_rates,
                                                      nb_objects_all_states,
                                                      sim_id, param_id):

    # get rates for all state transitions, depending on number of objects
    # in corresponding start state of transition
    transition_nb = 0
    total_rates[sim_id, param_id] = 0
    while transition_nb < transition_parameters.shape[0]:
        transition_states = all_transition_states[transition_nb]
        start_state = transition_states[0]
        transition_parameter = int(transition_parameters[transition_nb])
        transition_rate = parameter_value_array[transition_parameter,
                                                sim_id, param_id]
        if start_state == 0:
            # for state 0, the number of objects in state 0 does not matter
            current_transition_rates[transition_parameter,
                                     sim_id, param_id] = transition_rate
        else:
            nb_objects = nb_objects_all_states[int(start_state-1), sim_id, param_id]
            current_transition_rates[transition_nb,
                                     sim_id, param_id] = (nb_objects *
                                                          transition_rate)
        total_rates[sim_id, param_id] += current_transition_rates[transition_nb,
                                                                  sim_id, param_id]
        transition_nb += 1

def _get_random_number_cpu(sim_id, param_id, rng_states,
                           simulation_factor, parameter_factor):
    return np.random.rand()

def _get_random_number_gpu(sim_id, param_id,
                           rng_states, simulation_factor, parameter_factor):
    state = (sim_id+1)*simulation_factor + (param_id+1)*parameter_factor
    random_nb = cuda.random.xoroshiro128p_uniform_float32(rng_states,
                                                          state)
    return random_nb


def _get_times_of_next_transition(total_rates, reaction_times,
                                  sim_id, param_id, rng_states,
                                  simulation_factor, parameter_factor):
    # get time of next event for each simulation
    #
    number_threads = 128
    random_nb = _get_random_number(sim_id, param_id, rng_states,
                                   simulation_factor, parameter_factor)
    reaction_times[sim_id, param_id] = - (1/total_rates[sim_id, param_id] *
                                 math.log(random_nb))

def _determine_next_transition(total_rates, current_transition_rates,
                               current_transitions, sim_id, param_id,
                               rng_states, simulation_factor, parameter_factor):

    # set random number in zero rate positions to >1 to make threshold
    # higher than total rate, thereby preventing any reaction from
    # being executed
    if total_rates[sim_id, param_id] == 0:
        current_transitions[sim_id, param_id] = math.nan
        return

    # get which event happened in each simulation
    random_number = _get_random_number(sim_id, param_id, rng_states,
                                       simulation_factor, parameter_factor)

    threshold = total_rates[sim_id, param_id] * random_number

    # go through each transition and check whether it will occur
    current_rate_sum = 0
    transition_nb = 0
    current_transitions[sim_id, param_id] = math.nan
    while transition_nb < current_transition_rates.shape[0]:
        current_rate_sum += current_transition_rates[transition_nb,
                                                     sim_id, param_id]
        if current_rate_sum >= threshold:
            current_transitions[sim_id, param_id] = transition_nb
            break
        transition_nb += 1

    return


def _determine_positions_of_transitions(current_transitions,
                                        all_transition_states,
                                        nb_objects_all_states,
                                        all_transition_positions,
                                        object_states,
                                        sim_id, param_id, rng_states,
                                        simulation_factor, parameter_factor):

    # the transitions masks only tell which reaction happens in each
    # stimulation, but not which object in this simulation is affected
    # To get one random object of the possible objects,
    # first, create mask of index positions, so that each object for each
    # simulation has a unique identifier (index) within this simulation
    # setting all positions where no catastrophe can take place to 0
    transition_nb = current_transitions[sim_id, param_id]
    if math.isnan(transition_nb):
        all_transition_positions[sim_id, param_id] = math.nan
        return
    start_state = all_transition_states[int(transition_nb), 0]
    # if start state is 0, choose the first object with state 0
    if start_state == 0:
        object_pos = 0
        while object_pos < object_states.shape[0]:
            object_state = object_states[object_pos, sim_id, param_id]
            if object_state == 0:
                all_transition_positions[sim_id, param_id] = object_pos
                return
            object_pos += 1
    else:
        # for all other states, choose a random position with that state
        nb_objects = nb_objects_all_states[int(start_state-1), sim_id, param_id]
        random_object_pos = round(_get_random_number(sim_id, param_id,
                                                     rng_states,
                                                     simulation_factor,
                                                     parameter_factor)
                                     * nb_objects)
        object_pos = 0
        current_nb_state_objects = 0

        # go through all objects, check which one is in the start_state
        # and then choose the nth (n=random_object_pos) object that is in
        # the start_state
        while object_pos < object_states.shape[0]:
            object_state = object_states[object_pos, sim_id, param_id]
            if object_state == start_state:
                if current_nb_state_objects == random_object_pos:
                    all_transition_positions[sim_id, param_id] = object_pos
                    return
                current_nb_state_objects += 1
            object_pos += 1
    return None


def _execute_actions_on_objects(parameter_value_array, action_parameters,
                                action_state_array,
                                properties_array,
                                property_min_values,
                                property_max_values,
                                all_action_properties,
                                action_operation_array,
                                object_states,
                                reaction_times, sim_id, param_id):
    # execute actions on objects depending on state, before changing state
    action_nb = 0
    while action_nb < action_parameters.shape[0]:
        action_parameter = int(action_parameters[action_nb])
        # get the current action value, dependent on reaction time
        action_value = parameter_value_array[action_parameter, sim_id, param_id]
        current_action_value = action_value * reaction_times[sim_id, param_id]
        action_states = action_state_array[action_nb]
        action_properties = all_action_properties[action_nb]
        # action operation is -1 for subtracting and 1 for adding
        action_operation = action_operation_array[action_nb]
        # go through each object and check whether its state is in
        # action_states,
        object_pos = 0
        while object_pos < object_states.shape[0]:
            object_state = object_states[object_pos, sim_id, param_id]
            if object_state == 0:
                object_pos += 1
                continue
            is_target = False
            action_state_nb = 0
            if action_states[0] == 0:
                is_target = True
            else:
                while action_state_nb < len(action_states):
                    action_state = action_states[action_state_nb]
                    if object_state == action_state:
                        is_target = True
                        break
                    action_state_nb += 1
            # if the object has the correct state,
            # change the property values defined in the
            # action by the current action value
            # if action_nb == 1:
            #     print(is_target, action_properties)

            if is_target == False:
                object_pos += 1
                continue

            action_property_nb = 0
            while action_property_nb < len(action_properties):
                property_nb = action_properties[action_property_nb]
                property_val = properties_array[int(property_nb),
                                                 object_pos,
                                                 sim_id, param_id]
                new_property_val = (property_val +
                                    action_operation * current_action_value)
                # check whether property is below min
                # if there is just one non-nan value in min_value
                # then this is the min_value
                # if there are multiple non-nan values in min_value
                # then these are property numbers, except
                # for the first value, which is threshold
                # and the second value, which is the operation
                # (-1 for subtracting other property values from the current
                #  property value before comparing to the threshold,
                #  and +1 for adding other property values to the current
                #  property, before comparing to the threshold)
                min_value = property_min_values[int(property_nb)]
                threshold = min_value[0]
                if len(min_value) == 1:
                    threshold = threshold
                elif math.isnan(min_value[1]):
                    threshold = threshold
                else:
                    min_value_nb = 2
                    while min_value_nb < len(min_value):
                        val_property_nb = min_value[min_value_nb]
                        if math.isnan(val_property_nb):
                            break
                        # if property values are added, the threshold should
                        # be reduced by the other property values
                        # (since higher other property values means that the
                        # threshold "is reached earlier")
                        if min_value[1] == 1:
                            threshold = threshold - properties_array[int(val_property_nb),
                                                                     object_pos,
                                                                     sim_id,
                                                                     param_id]
                        if min_value[1] == -1:
                            threshold = threshold + properties_array[int(val_property_nb),
                                                                     object_pos,
                                                                     sim_id,
                                                                     param_id]
                        min_value_nb += 1
                if new_property_val < threshold:
                    new_property_val = threshold

                # check whether the property value is above the max val
                # similarly as for the min val condition

                max_value = property_max_values[int(property_nb)]
                threshold = max_value[0]
                if len(max_value) == 1:
                    threshold = threshold
                elif math.isnan(max_value[1]):
                    threshold = threshold
                else:
                    max_value_nb = 2
                    while max_value_nb < len(max_value):
                        val_property_nb = max_value[max_value_nb]
                        if math.isnan(val_property_nb):
                            break
                        if max_value[1] == 1:
                            threshold = threshold - properties_array[int(val_property_nb),
                                                                     object_pos,
                                                                     sim_id,
                                                                     param_id]
                        if max_value[1] == -1:
                            threshold = threshold + properties_array[int(val_property_nb),
                                                                     object_pos,
                                                                     sim_id,
                                                                     param_id]
                        max_value_nb += 1
                if new_property_val > threshold:
                    new_property_val = threshold
                # set the property val to the property val within the
                # [min_value, max_value] limits
                properties_array[int(property_nb),
                                 object_pos,
                                 sim_id, param_id] = new_property_val

                action_property_nb += 1
            object_pos += 1
        action_nb += 1

    return


def _update_object_states(current_transitions, all_transition_states,
                          all_transition_positions, object_states,
                          nb_objects_all_states, properties_array,
                          all_transition_tranferred_vals,
                          all_transition_set_to_zero_properties,
                          property_start_vals,
                          sim_id, param_id,
                          rng_states, simulation_factor, parameter_factor):
    # update the simulations according to executed transitions
    transition_number = current_transitions[sim_id, param_id]
    if math.isnan(transition_number):
        return
    transition_states = all_transition_states[int(transition_number)]
    start_state = transition_states[0]
    end_state = transition_states[1]
    transition_position = all_transition_positions[sim_id, param_id]
    object_states[int(transition_position),sim_id, param_id] = end_state
    # change the object counter according to the executed transition
    if start_state != 0:
        nb_objects_all_states[int(start_state)-1, sim_id, param_id] -= 1
    if end_state != 0:
        nb_objects_all_states[int(end_state)-1, sim_id, param_id] += 1
    # change property values based on transitions
    if start_state == 0:
        # if a new object was created, set property values according to
        # defined value
        property_nb = 0
        while property_nb < property_start_vals.shape[0]:
            property_start_val = property_start_vals[property_nb]
            # if there is only one non-nan value or just one value in total
            # then the first value is the actual value that the property
            # should be set to
            if len(property_start_val) == 1:
                property_val = property_start_val[0]
            elif math.isnan(property_start_val[1]):
                property_val = property_start_val[0]
            else:
                # if there are two non-nan start property vals, then
                # the property val should be a random number between these
                # two numbers, with the first number indicating the start
                # (lower value) and the second number the end of the
                # interval (higher value)
                random_nb = _get_random_number(sim_id, param_id, rng_states,
                                               simulation_factor,
                                               parameter_factor)
                # to get a random number in that range take a random nb
                # between 0 and 1 and multiply it with the range of the
                # interval, then add the start point of the interval
                # to have the minimum of the random value at the start of
                # the interval
                interval_range = (property_start_val[1] -
                                  property_start_val[0])
                property_val = (random_nb * interval_range +
                                property_start_val[0])
            properties_array[property_nb,
                             int(transition_position),
                             sim_id, param_id] = property_val

            property_nb += 1

    elif end_state == 0:
        # if an object was removed, set property values to NaN
        property_nb = 0
        while property_nb < properties_array.shape[0]:
            properties_array[property_nb,
                             int(transition_position),
                             sim_id, param_id] = math.nan
            property_nb += 1

    # if the object is not removed, it might be that the transition triggers
    # other events as well
    if end_state != 0:
        # if property values are transfered, then the first number
        # is the property number source and the second is the target
        # if the value is nan then there is no transfer
        transfered_vals = all_transition_tranferred_vals[int(transition_number)]

        if math.isnan(transfered_vals[0]) == False:
            source_property_number = transfered_vals[0]
            target_property_number = transfered_vals[1]
            source_val = properties_array[int(source_property_number),
                                          int(transition_position),
                                          sim_id, param_id]
            target_val = properties_array[int(target_property_number),
                                          int(transition_position),
                                          sim_id, param_id]
            properties_array[int(target_property_number),
                             int(transition_position),
                             sim_id, param_id] = source_val

            properties_array[int(source_property_number),
                             int(transition_position),
                             sim_id, param_id] = 0

        # if properties should be set to zero for the current transition
        # do that at the current position
        set_to_zero_properties = all_transition_set_to_zero_properties[
            int(transition_number)]
        if math.isnan(set_to_zero_properties[0]) == False:
            zero_property_nb = 0
            while zero_property_nb < len(set_to_zero_properties):
                zero_property = set_to_zero_properties[zero_property_nb]
                if math.isnan(zero_property):
                    break
                properties_array[int(zero_property),
                                 int(transition_position),
                                 sim_id, param_id] = 0
                zero_property_nb += 1

    return None

def _remove_objects(all_object_removal_properties, object_removal_operations,
                    object_states, properties_array, sim_id, param_id):
    removal_nb = 0
    while removal_nb < len(all_object_removal_properties):
        # for object removal, there are two array with
        # several values for each condition
        # in all_object_removal_properties
        # the first value is the operation used to combine property values
        # with 1 meaning summing and -1 meaning subtracting
        # for subtracting all properties after the first one are subtracted
        # from the first one
        # the following values are property numbers
        object_removal_properties = all_object_removal_properties[removal_nb]
        property_operation = object_removal_properties[0]
        properties = object_removal_properties[1:]
        # for the second array, object_removal_operations
        # the first value is the operation, which can be 1 for the combined
        # property value is larger then a threshold
        # and -1 for the combined property value is smaller then a threshold
        # the second value is the threshold
        object_removal_operation = object_removal_operations[removal_nb]
        threshold_operation = object_removal_operation[0]
        threshold = object_removal_operation[1]

        object_pos = 0
        while object_pos < object_states.shape[0]:
            # combine property values according to property_operation
            combined_property_vals = properties_array[int(properties[0]),
                                                      int(object_pos),
                                                      sim_id, param_id]
            if math.isnan(combined_property_vals):
                object_pos += 1
            else:
                property_idx = 1
                while property_idx < properties.shape[0]:
                    property_nb = int(properties[property_idx])
                    property_val = properties_array[property_nb,
                                                    int(object_pos),
                                                    sim_id, param_id]
                    combined_property_vals += property_operation * property_val
                    property_idx += 1
                # check whether combined property values
                # are above or below threshold
                if threshold_operation == -1:
                    remove_object = combined_property_vals > threshold
                elif threshold_operation == 1:
                    remove_object = combined_property_vals < threshold

                # if object should be removed, set state to 0 and properties to NaN
                if remove_object:
                    object_states[int(object_pos),
                                  sim_id, param_id] = 0
                    property_nb = 0
                    while property_nb < properties_array.shape[0]:
                        properties_array[property_nb,
                                         int(object_pos),
                                         sim_id, param_id] = math.nan

                        property_nb += 1
                object_pos += 1

        removal_nb += 1

def _save_values_with_temporal_resolution(timepoint_array,
                                          object_state_time_array,
                                         properties_time_array,
                                         current_timepoint_array, times,
                                         object_states, properties_array,
                                         time_resolution, save_initial_state,
                                          sim_id, param_id):
    # check .if the next timepoint was reached, then save all values
    # at correct position
    current_timepoint = current_timepoint_array[sim_id, param_id]
    if times[0, sim_id, param_id] >= (current_timepoint * time_resolution[0] +
                               time_resolution[0]):
        time_jump = math.floor((times[0,sim_id, param_id] -
                                     (current_timepoint * time_resolution[0]))
                                    / time_resolution[0])
        current_timepoint += time_jump

        current_timepoint_array[sim_id, param_id] = current_timepoint
        # if the initial state was not saved then the index is actually the
        # timepoint minus 1
        if save_initial_state:
            timepoint_idx = int(current_timepoint)
        else:
            timepoint_idx = int(current_timepoint) - 1

        timepoint_array[timepoint_idx,
                        sim_id, param_id] = current_timepoint
        # copy all current data into time-resolved data
        object_pos = 0
        while object_pos < object_states.shape[0]:
            object_state = object_states[object_pos, sim_id, param_id]

            object_state_time_array[timepoint_idx,
                                    object_pos,
                                    sim_id, param_id] = object_state

            property_nb = 0
            while property_nb < properties_array.shape[0]:
                property_val = properties_array[property_nb,
                                                object_pos, sim_id, param_id]
                properties_time_array[timepoint_idx,
                                      property_nb,
                                      object_pos,
                                      sim_id, param_id] = property_val
                property_nb += 1
            object_pos += 1