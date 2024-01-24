import numpy as np

# NUMBA_DEBUG=1
# NUMBA_DEVELOPER_MODE = 1
# NUMBA_DEBUGINFO = 1
import numba
import numba.cuda.random
import math
from numba import cuda

from matplotlib import pyplot as plt


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
                            changed_start_values_array,
                            creation_on_objects,
                        some_creation_on_objects,
                        all_object_removal_properties,
                        object_removal_operations,

                        nb_objects_all_states,
                        total_rates, reaction_times,
                        current_transitions,
                        all_transition_positions,

                            # all parameters for nucleation on objects
                            end_position_tmax_array,
                            properties_tmax_array,
                            properties_tmax, properties_tmax_sorted,
                            current_sum_tmax,
                            property_changes_tmax_array,
                            property_changes_tmin_array,
                            property_changes_per_state,
                                    nucleation_changes_per_state,
                            total_property_changes,

                            tau_square_eq_constant,
                            tau_square_eq_second_order,

                            highest_idx_with_object,lowest_idx_no_object,

                        current_timepoint_array, timepoint_array,
                        object_state_time_array, properties_time_array,
                        time_resolution, min_time, save_initial_state,

                        nb_parallel_cores, thread_masks,

                            local_density, local_resolution,total_density,
                            _, rng_states, simulation_factor, parameter_factor
                        ):
    # np.random.seed(seed)
    iteration_nb = 0

    # for multiprocessing,
    # have one array with the current status of each sim_id, param_id
    # combination (0 for not done, 1 for done): sim_status
    # also have an array keeping track of the current number of parallel
    # processes: nb_parallel_cores
    # when a simulation is finished and current_sim_nb + nb_processes
    # is larger than total_nb_simulations, add the core to another simulation

    # The threads participating in a simulation are kept track of with an int32
    # mask for the corresponding warp (currently always 32 threads within a
    # block in numba)

    # number of simulations should be a multiple of the warp size (32)
    # and the warp size should be whole number divisable by nb_parallel_cores
    # (1, 2, 4, 8, 16, 32 allowed)

    # mask = 0
    # mask |= (1<<31)
    # print(mask)
    #
    # times[0, 0, 0] = mask
    #
    # print(times[0, 0, 0])
    #

    # a = int(times[0,0,0])
    #
    # a &=~ (1<<31)
    # a |= (1<<31)

    nb_processes = cuda.gridsize(1)
    thread_id = cuda.grid(1)
    grid = cuda.cg.this_grid()

    total_nb_simulations = (nb_simulations * nb_parameter_combinations *
                            nb_parallel_cores)
    # print(cuda.gridsize(1), total_nb_simulations)
    current_sim_nb = thread_id
    new_simulation = False
    while current_sim_nb < total_nb_simulations:
        # For each parameter combination the defined number of simulations
        # are done on a defined number of cores
        if not new_simulation:
            param_id = int(math.floor(current_sim_nb /
                                      (nb_simulations * nb_parallel_cores)))
            sim_id = int(math.floor((current_sim_nb -
                                     param_id * nb_simulations * nb_parallel_cores)
                                    / nb_parallel_cores))
            core_id = int(current_sim_nb -
                          param_id * nb_simulations * nb_parallel_cores -
                          sim_id * nb_parallel_cores)

            warp_thread_idx = cuda.laneid
            cuda.atomic.add(thread_masks, (sim_id, param_id),
                            (1 << warp_thread_idx))

            # warp_mask = int(thread_masks[sim_id, param_id])
            # warp_mask |= (1<<warp_thread_idx)
            # thread_masks[sim_id, param_id] = warp_mask

            if some_creation_on_objects & (core_id == 0):
                nucleation_on_objects_rate = _get_nucleation_on_objects_rate(creation_on_objects,
                                                                            transition_parameters,
                                                                            parameter_value_array,
                                                                             core_id, sim_id, param_id)

                _get_property_action_changes(property_changes_per_state,
                                                total_property_changes,
                                                action_parameters,
                                               parameter_value_array,
                                                action_state_array,
                                               all_action_properties,
                                                action_operation_array,
                                               core_id, sim_id, param_id)

                _get_nucleation_changes_per_state(nucleation_changes_per_state,
                                                      property_changes_per_state,
                                                      nucleation_on_objects_rate,
                                                  core_id, sim_id, param_id)

            if nb_parallel_cores > 1:
                cuda.syncwarp(int(thread_masks[sim_id, param_id]))

        while True:
            # # wait for old and new threads to be synced here
            # cuda.syncwarp(new_threads_masks[sim_id, param_id])
            
            success = _run_iteration(object_states,
                                     properties_array,
                                   times, parameter_value_array,
                                   transition_parameters, all_transition_states,
                                   action_parameters, action_state_array,
                                   all_action_properties, action_operation_array,
                                   current_transition_rates,
                                   property_start_values,
                                   property_min_values, property_max_values,
                                   all_transition_tranferred_vals,
                                   all_transition_set_to_zero_properties,
                                    changed_start_values_array,
                                    creation_on_objects,
                                some_creation_on_objects,
                                   all_object_removal_properties,
                                   object_removal_operations,

                                   nb_objects_all_states,
                                   total_rates, reaction_times,
                                   current_transitions,
                                   all_transition_positions,

                                     # all parameters for nucleation on objects
                                     end_position_tmax_array,
                                     properties_tmax_array,
                                     properties_tmax, properties_tmax_sorted,
                                     current_sum_tmax,
                                     property_changes_tmax_array,
                                     property_changes_tmin_array,
                                     property_changes_per_state,
                                    nucleation_changes_per_state,
                                     total_property_changes,

                                     tau_square_eq_constant,
                                     tau_square_eq_second_order,

                            highest_idx_with_object,lowest_idx_no_object,

                           current_timepoint_array, timepoint_array,
                           object_state_time_array, properties_time_array,
                           time_resolution, save_initial_state,

                                     nb_parallel_cores, thread_masks,

                            local_density, local_resolution,total_density,

                           rng_states, core_id, sim_id, param_id,
                                     simulation_factor, parameter_factor,

                           )
            if (current_timepoint_array[sim_id, param_id] >=
                    math.floor(min_time/time_resolution[0])):
                # print(iteration_nb)
                break
            if success == 0:
                break
            # if iteration_nb == 500:
            #     break
            iteration_nb += 1
        if success == 0:
            time_idx = 0
            while time_idx < timepoint_array.shape[0]:
                timepoint_array[time_idx, sim_id, param_id] = math.nan
                time_idx += 1
            break
        current_sim_nb += nb_processes
        # if all simulations for this process were done, reassign to another
        # simulation
        # if current_sim_nb >= total_nb_simulations:
        #     # Reassign process to simulation of a process in the same warp
        #     warp_nb = thread_id // cuda.warpsize
        #     warp_thread_idx = thread_id - warp_nb * cuda.warpsize
        #     current_thread_nb = warp_nb * cuda.warpsize
        #     last_thread = (warp_nb + 1) * cuda.warpsize
        #     while current_thread_nb < last_thread:
        #         # don't consider switching to own thread_id
        #         if current_thread_nb != thread_id:
        #             current_sim_nb = thread_to_sim_id[current_thread_nb, 0]
        #             current_param_nb = thread_to_sim_id[current_thread_nb, 1]
        #             # make sure that process is added to a different simulation
        #             if ((current_sim_nb != sim_id) &
        #                     (current_param_nb != param_id)):
        #                 new_sim_id = current_sim_nb
        #                 new_param_id = current_param_nb
        #                 # update sim id of current thread
        #                 thread_to_sim_id[thread_id, 0] = new_sim_id
        #                 thread_to_sim_id[thread_id, 1] = new_param_id
        #                 thread_to_sim_id[thread_id, 2] = new_param_id
        #                 # update current sim nb based on new sim id
        #                 current_sim_nb = (new_param_id * nb_simulations *
        #                                   nb_parallel_cores +
        #                                   new_sim_id * nb_parallel_cores)
        #                 # no add one to the number of parallel cores of the
        #                 # new sim id
        #                 # This returns the number of cores before new ones were
        #                 # added
        #                 old_nb_cores = cuda.atomic.add(nb_parallel_cores,
        #                                                (new_sim_id, new_param_id), 1)
        #                 # How to get the new core_id, considering that
        #                 # multiple new cores could be added
        #                 # the number of added cores is the nb of parallel cores
        #                 # of the old sim id
        #                 # the mask of the old sim id defines which
        #                 # warp thread idx were added
        #                 # first sync newly added cores
        #                 cuda.syncwarp(thread_masks[sim_id, param_id])
        #                 # then go through all threads in the warp and
        #                 # the first nan core ids will get the smaller new
        #                 # core_ids
        #                 new_core_id = old_nb_cores - 1
        #                 new_core_id_thread_nb = warp_nb * cuda.warpsize
        #                 while new_core_id_thread_nb < last_thread:
        #                     # for each not assigned core id in the warp
        #                     # increase the core id by one
        #                     if math.isnan(thread_to_sim_id[new_core_id_thread_nb,
        #                                                    2]):
        #                         new_core_id += 1
        #                     # if the new core id thread number in the loop
        #                     # is the thread number of the active process
        #                     # then assign the core id
        #                     if new_core_id_thread_nb == thread_id:
        #                         core_id = new_core_id
        #                         break
        #                     new_core_id_thread_nb += 1
        #                 cuda.syncwarp(thread_masks[sim_id, param_id])
        #                 thread_to_sim_id[thread_id, 2] = core_id
        #                 # create new mask by adding the mask of the old sim id
        #                 # and the new sim id
        #                 new_threads_masks[new_sim_id,
        #                                   new_param_id] = (thread_masks[sim_id,
        #                                                                 param_id]
        #                                                    +
        #                                                    thread_masks[new_sim_id,
        #                                                                 new_param_id])
        #                 sim_id = new_sim_id
        #                 param_id = new_param_id
        #                 new_simulation = True
        #                 break
        #         current_thread_nb += 1

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
                            changed_start_values_array,
                            creation_on_objects,
                        some_creation_on_objects,
                        all_object_removal_properties,
                        object_removal_operations,

                        nb_objects_all_states,
                        total_rates, reaction_times,
                        current_transitions,
                        all_transition_positions,

                            # all parameters for nucleation on objects
                            end_position_tmax_array,
                            properties_tmax_array,
                            properties_tmax, properties_tmax_sorted,
                            current_sum_tmax,
                            property_changes_tmax_array,
                            property_changes_tmin_array,
                            property_changes_per_state,
                            nucleation_changes_per_state,
                            total_property_changes,

                            tau_square_eq_constant,
                            tau_square_eq_second_order,

                            highest_idx_with_object,lowest_idx_no_object,

                        current_timepoint_array, timepoint_array,
                        object_state_time_array, properties_time_array,
                        time_resolution, min_time, save_initial_state,

                            nb_parallel_cores, thread_masks,

                            local_density, local_resolution,total_density,
                            seed, rng_states = None
                        ):
    np.random.seed(seed)
    iteration_nb = 0

    thread_id = 0
    core_id = 0

    # For each parameter combination the defined number of simulations are done
    param_id = int(math.floor(thread_id / nb_simulations))
    sim_id = int(thread_id - param_id * nb_simulations)
    if some_creation_on_objects & (core_id == 0):
        nucleation_on_objects_rate = _get_nucleation_on_objects_rate(
                                                                creation_on_objects,
                                                                transition_parameters,
                                                                parameter_value_array,
                                                                core_id, sim_id,
                                                                param_id)

        _get_property_action_changes(
                                    property_changes_per_state,
                                    total_property_changes,
                                    action_parameters,
                                    parameter_value_array,
                                    action_state_array,
                                    all_action_properties,
                                    action_operation_array,
                                    core_id, sim_id, param_id)

        _get_nucleation_changes_per_state(nucleation_changes_per_state,
                                          property_changes_per_state,
                                          nucleation_on_objects_rate,
                                          core_id, sim_id, param_id)


    assertion_checks = True
    if thread_id < (nb_simulations * nb_parameter_combinations):
        last_transition = 0
        while True:
            success = _run_iteration(object_states,
                                     properties_array,
                                       times, parameter_value_array,
                                       transition_parameters,
                                     all_transition_states,
                                       action_parameters, action_state_array,
                                       all_action_properties,
                                     action_operation_array,
                                       current_transition_rates,
                                       property_start_values,
                                       property_min_values, property_max_values,
                                       all_transition_tranferred_vals,
                                       all_transition_set_to_zero_properties,
                                        changed_start_values_array,
                                        creation_on_objects,
                                    some_creation_on_objects,
                                       all_object_removal_properties,
                                       object_removal_operations,

                                       nb_objects_all_states,
                                       total_rates, reaction_times,
                                       current_transitions,
                                       all_transition_positions,

                                     # all parameters for nucleation on objects
                                     end_position_tmax_array,
                                     properties_tmax_array,
                                     properties_tmax, properties_tmax_sorted,
                                     current_sum_tmax,
                                     property_changes_tmax_array,
                                     property_changes_tmin_array,
                                     property_changes_per_state,
                                    nucleation_changes_per_state,
                                     total_property_changes,

                                     tau_square_eq_constant,
                                     tau_square_eq_second_order,

                                     highest_idx_with_object,lowest_idx_no_object,

                                       current_timepoint_array, timepoint_array,
                                       object_state_time_array, 
                                     properties_time_array,
                                       time_resolution, save_initial_state,

                                     nb_parallel_cores, thread_masks,
            
                                       local_density, local_resolution,
                                     total_density,
                                       rng_states, core_id, sim_id, param_id
                           )

            if (current_timepoint_array[sim_id, param_id] >=
                    math.floor(min_time/time_resolution[0])):
                break

            print(iteration_nb, times[sim_id, param_id])

            if success == 0:
                print("\n no success!")
                break

            if assertion_checks:
                # print("\n")
                # print(nb_objects_all_states[:,0,0])
                for state in range(1,np.max(object_states)+1):
                    state_mask = object_states == state
                    # if state == 1:
                    #     plus_pos = (properties_array[0][state_mask]
                    #                 + properties_array[1][state_mask])
                    #     # print(current_transitions)
                    #     print(properties_array[0][state_mask][plus_pos == 20])

                    plus_pos = (properties_array[0][state_mask]
                                + properties_array[1][state_mask]
                                + properties_array[2][state_mask])
                    # print(current_transitions)
                    # print("\n", state)
                    # print(properties_array[2][state_mask][plus_pos == 20])
                    # check that no property value of an object is nan
                    for property_nb in range(properties_array.shape[0]):
                        property_vals = properties_array[property_nb]
                        property_vals_state = property_vals[state_mask]
                        nb_nan = len(property_vals_state[
                                         np.isnan(property_vals_state)])
                        # if (property_nb > 1):# | (property_nb == 0):
                        #     if (state >= 1):#| (state == 5):
                        #         if len(property_vals_state) > 0:
                        #             print(iteration_nb, state, property_nb)
                        #             print(property_vals_state)
                        # # make sure that there are no stable MTs
                        # # with length 0
                        if (state == 3) & (property_nb == 2):
                            nb_zeros = len(property_vals_state[
                                               property_vals_state == 0])
                            assert nb_zeros == 0

                        # if (state == 3) & (property_nb == 1):
                        #             print(property_vals_state)
                    # check that the calculated number of objects is correct
                    nb_objects = len(object_states[object_states == state])
                    saved_nb_objects = nb_objects_all_states[state-1]
                    assert nb_objects == saved_nb_objects
                    # check that the maximum transition position is within
                    # the size of the array
                    max_position = all_transition_positions.max()
                    assert max_position < object_states.shape[0]
            last_transition = int(current_transitions[0])

            iteration_nb += 1
        if success == 0:
            time_idx = 0
            while time_idx < timepoint_array.shape[0]:
                timepoint_array[time_idx, sim_id, param_id] = math.nan
                time_idx += 1


def _run_iteration(object_states, properties_array, times,
                   parameter_value_array,
                   transition_parameters, all_transition_states,
                   action_parameters, action_state_array,
                   all_action_properties, action_operation_array,
                   current_transition_rates,
                   property_start_values,
                   property_min_values, property_max_values,
                   all_transition_tranferred_vals,
                   all_transition_set_to_zero_properties,
                    changed_start_values_array,
                    creation_on_objects,
                        some_creation_on_objects,
                   all_object_removal_properties,
                   object_removal_operations,

                   nb_objects_all_states,
                   total_rates, reaction_times,
                   current_transitions,
                   all_transition_positions,

                   # all parameters for nucleation on objects
                   end_position_tmax_array, properties_tmax_array,
                   properties_tmax, properties_tmax_sorted,
                   current_sum_tmax,
                   property_changes_tmax_array, property_changes_tmin_array,
                   property_changes_per_state, nucleation_changes_per_state,
                   total_property_changes,

                   tau_square_eq_constant,
                   tau_square_eq_second_order,

                   highest_idx_with_object, lowest_idx_no_object,

                   current_timepoint_array, timepoint_array,
                   object_state_time_array, properties_time_array,
                   time_resolution, save_initial_state,

                   nb_parallel_cores, thread_masks,

                   local_density, local_resolution, total_density,

                   rng_states, core_id, sim_id, param_id,
                   simulation_factor=None, parameter_factor=None
                   ):

    grid = cuda.cg.this_grid()
    # create tensor for x (position in neurite), l (length of microtubule)
    # and time
    # start = time.time()
    # UPDATE nb_objects_all_states in each iteration at each iteration
    # thereby there is no need to go through all object states


    if some_creation_on_objects:

        _reset_local_density(local_density, nb_parallel_cores, core_id,
                             sim_id, param_id)
        if nb_parallel_cores > 1:
            cuda.syncwarp(int(thread_masks[sim_id, param_id]))

        _get_local_and_total_density(local_density,
                                           total_density,
                                           local_resolution,
                                           highest_idx_with_object,
                                           properties_array,
                                           object_states,
                                           nb_parallel_cores,grid,thread_masks,
                                           core_id, sim_id, param_id)

        if nb_parallel_cores > 1:
            cuda.syncwarp(int(thread_masks[sim_id, param_id]))

    # increase performance through an array for each object state
    # of which transitions are influenced by it
    # then don't recalculate the transition rates each iteration
    # but just update the few rates affected by the changed object state
    _get_rates = _get_total_and_single_rates_for_state_transitions
    _get_rates(parameter_value_array, transition_parameters,
               all_transition_states, current_transition_rates, total_rates,
               nb_objects_all_states,creation_on_objects,
               total_density, local_resolution,
               nb_parallel_cores, grid, thread_masks, core_id, sim_id, param_id)

    if nb_parallel_cores > 1:
        cuda.syncwarp(int(thread_masks[sim_id, param_id]))

    if some_creation_on_objects:
        # nucleation rates for nucleation on objects are time-dependent
        # since objects might increase or decrease in size and might move out
        # of the compartment. This will change density, independent of
        # transitions, since it will happen through actions
        # Go through each object and check the maximum time that the object can
        # execute the action (e.g. before leaving the compartment or hitting the
        # end of the compartment).

        # property_tmax is array with dimensions (properties)
        # array is filled with nan by default since not all properties
        # have tmax values

        # then iterate through each object and then through each property net
        # change
        # NOT CONSIDERED:
        # - object vanishing due to reducing all properties values to 0
        # - no object removal when end below 0!
        _get_tmin_tmax_for_property_changes(property_changes_tmax_array,
                                             property_changes_tmin_array,
                                             property_changes_per_state,
                                             total_property_changes,
                                               current_sum_tmax,
                                               properties_tmax_sorted,
                                               properties_tmax,
                                               properties_tmax_array,
                                               end_position_tmax_array,

                                               highest_idx_with_object,
                                               object_states, properties_array,
                                                property_max_values,
                                             nb_parallel_cores,grid,
                                               core_id, sim_id, param_id
                                               )

        if nb_parallel_cores > 1:
            cuda.syncwarp(int(thread_masks[sim_id, param_id]))

        # Then iterate through possible tau values, which lead to different
        # included factors. Start with exponential divided by total baseline
        # rate. Calculate tau, if different from initial guess, use calculated
        # tau as new tau. Repeat loop until calculated tau and guessed tau
        # differ < 1%.

        # To calculate tau, check for each object and action whether the tau
        # is within the minimum and maximum, below the minimum or above the
        # maximum.

        # The equation to be solved for tau is
        # rand_exp = Integral, from 0 to tau, of sum of all rates
        # Since all actions lead to a linear change of density over time
        # the equation to solve to get the reaction time is a quadratic equation.
        # rand_exp is a number drawn from the standard exponential distribution.
        # The negative of this number contributes to the constant term of this
        # equation.
        # For this equation, objects on which actions are executed until the end
        # of tau contribute to the squared term (since they include a direct
        # dependence on tau). Objects on which actions are executed but not
        # until the end of tau contribute to the constant term since there is no
        # dependence on tau at all. All rates that are not time dependent
        # contribute to the first order term term since the integral adds the
        # dependence on tau.

        # To get the reaction rate with time dependent rates, draw a number from
        # the standard exponential distribution. The negative of this number
        # contributes to the constant term of the quadratic equation.

        # The variables for quadratic equation are:
        tau = _get_tau(total_rates, highest_idx_with_object, object_states,
                        property_changes_per_state,nucleation_changes_per_state,
                        property_changes_tmin_array,property_changes_tmax_array,
                    tau_square_eq_constant, tau_square_eq_second_order,
                         rng_states, simulation_factor, parameter_factor,
                         nb_parallel_cores, grid, thread_masks,
                       core_id, sim_id, param_id)

        reaction_times[sim_id, param_id] = tau

        if nb_parallel_cores > 1:
            cuda.syncwarp(int(thread_masks[sim_id, param_id]))

    else:

        get_reaction_times = _get_times_of_next_transition
        get_reaction_times(total_rates, reaction_times, sim_id, param_id,
                           rng_states, simulation_factor, parameter_factor)

    # check if a new timepoint should be saved
    # before updating object states and executing actions
    current_timepoint = current_timepoint_array[sim_id, param_id]
    next_timepoint = (current_timepoint * time_resolution[0] +
                      time_resolution[0])

    if ((times[0, sim_id, param_id] +
        reaction_times[sim_id, param_id]) >= next_timepoint):
        # if a new timepoint should be saved, the current transition went beyond
        # that timepoint, which is particularly a problem for simulations with
        # a low number of object in which the time from one to the next simulation
        # can be quite high. In that case the time passed beyond the timepoint for
        # saving can also be high.
        # Therefore once a new timepoint should be saved, don't execute transition
        # instead execute the actions until the timepoint for saving. Then change
        # reaction times to the time from the saved timepoint until the transition
        # timepoint and execute actions and transitions normally.
        reaction_time_tmp = next_timepoint - times[0, sim_id, param_id]

        _execute_actions_on_objects(parameter_value_array, action_parameters,
                                    action_state_array,
                                    properties_array,
                                    property_min_values,
                                    property_max_values,
                                    all_action_properties,
                                    action_operation_array,
                                    object_states,
                                    reaction_time_tmp,
                                    highest_idx_with_object,
                                    nb_parallel_cores, grid, core_id,
                                    sim_id, param_id)

        if nb_parallel_cores > 1:
            cuda.syncwarp(int(thread_masks[sim_id, param_id]))

        if core_id == 0:
            times[0, sim_id, param_id] = (times[0, sim_id, param_id] +
                                          reaction_time_tmp)

        if nb_parallel_cores > 1:
            cuda.syncwarp(int(thread_masks[sim_id, param_id]))

        _save_values_with_temporal_resolution(timepoint_array,
                                              object_state_time_array,
                                              properties_time_array,
                                              current_timepoint_array, times,
                                              object_states, properties_array,
                                              time_resolution, save_initial_state,
                                              highest_idx_with_object,
                                              nb_parallel_cores, grid,
                                              thread_masks, core_id,
                                              sim_id, param_id)

        if core_id == 0:
            reaction_times[sim_id, param_id] = (reaction_times[sim_id, param_id] -
                                                reaction_time_tmp)

        if nb_parallel_cores > 1:
            cuda.syncwarp(int(thread_masks[sim_id, param_id]))

    _execute_actions_on_objects(parameter_value_array, action_parameters,
                                action_state_array,
                                properties_array,
                                property_min_values,
                                property_max_values,
                                all_action_properties,
                                action_operation_array,
                                object_states,
                                reaction_times[sim_id, param_id],
                                highest_idx_with_object,
                                nb_parallel_cores, grid, core_id,
                                sim_id, param_id)

    if nb_parallel_cores > 1:
        cuda.syncwarp(int(thread_masks[sim_id, param_id]))

    if some_creation_on_objects:

        _reset_local_density(local_density, nb_parallel_cores, core_id,
                             sim_id, param_id)
        if nb_parallel_cores > 1:
            cuda.syncwarp(int(thread_masks[sim_id, param_id]))

        # Update local and total density based on deterined tau
        _get_local_and_total_density(local_density,
                                       total_density,
                                       local_resolution,
                                       highest_idx_with_object,
                                       properties_array,
                                       object_states,
                                       nb_parallel_cores, grid,thread_masks,
                                       core_id,
                                       sim_id, param_id)

        if nb_parallel_cores > 1:
            cuda.syncwarp(int(thread_masks[sim_id, param_id]))

        # then get updated rates to have the correct nucleation rate and
        # total rate
        _get_rates = _get_total_and_single_rates_for_state_transitions
        _get_rates(parameter_value_array, transition_parameters,
                   all_transition_states, current_transition_rates, total_rates,
                   nb_objects_all_states, creation_on_objects,
                   total_density, local_resolution,
                   nb_parallel_cores, grid, thread_masks,
                   core_id, sim_id, param_id)

    if core_id == 0:
        _determine_next_transition(total_rates, current_transition_rates,
                                   current_transitions, sim_id, param_id,
                                   rng_states, simulation_factor, parameter_factor)
    if nb_parallel_cores > 1:
        cuda.syncwarp(int(thread_masks[sim_id, param_id]))

    if core_id == 0:
        # speed up searching for xth object with correct state
        # by keeping track of the position of each object in each state
        _determine_positions_of_transitions(current_transitions,
                                            all_transition_states,
                                            nb_objects_all_states,
                                            all_transition_positions,
                                            object_states, core_id,
                                            sim_id, param_id, rng_states,
                                            highest_idx_with_object,
                                            lowest_idx_no_object,
                                            simulation_factor, parameter_factor)
    if nb_parallel_cores > 1:
        cuda.syncwarp(int(thread_masks[sim_id, param_id]))

    if math.isnan(all_transition_positions[sim_id, param_id]):
        return 0

    _update_object_states(current_transitions, all_transition_states,
                          all_transition_positions, object_states,
                          nb_objects_all_states,
                          properties_array,
                          all_transition_tranferred_vals,
                          all_transition_set_to_zero_properties,
                          property_start_values,
                            changed_start_values_array,
                            creation_on_objects,
                          highest_idx_with_object,lowest_idx_no_object,

                          local_density, total_density,
                          local_resolution,
                          nb_parallel_cores, grid, core_id,
                          sim_id, param_id,
                          rng_states, simulation_factor, parameter_factor)

    if nb_parallel_cores > 1:
       cuda.syncwarp(int(thread_masks[sim_id, param_id]))

    _remove_objects(all_object_removal_properties, object_removal_operations,
                    nb_objects_all_states, object_states, properties_array,
                    highest_idx_with_object, lowest_idx_no_object,
                    nb_parallel_cores, grid, core_id, sim_id, param_id)

    if core_id == 0:
        times[0, sim_id, param_id] = (times[0, sim_id, param_id] +
                                      reaction_times[sim_id, param_id])

    if nb_parallel_cores > 1:
       cuda.syncwarp(int(thread_masks[sim_id, param_id]))

    return 1

def _get_random_number():
    pass

def _decorate_all_functions_for_cpu():

    # global _get_random_number
    # _get_random_number = _get_random_number_cpu

    global _get_random_number
    if not isinstance(_get_random_number,
                      numba.core.registry.CPUDispatcher):
        _get_random_number = numba.njit(_get_random_number_cpu)

    # global _execute_simulation_cpu
    # if not isinstance(_execute_simulation_cpu,
    #                   numba.core.registry.CPUDispatcher):
    #     _execute_simulation_cpu = numba.njit(_execute_simulation_cpu)


    global _run_iteration
    if not isinstance(_run_iteration,
                      numba.core.registry.CPUDispatcher):
        _run_iteration = numba.njit(_run_iteration)

    global _get_nucleation_on_objects_rate
    if not isinstance(_get_nucleation_on_objects_rate,
                      numba.core.registry.CPUDispatcher):
        _get_nucleation_on_objects_rate = numba.njit(
            _get_nucleation_on_objects_rate)

    global _get_nucleation_changes_per_state
    if not isinstance(_get_nucleation_changes_per_state,
                      numba.core.registry.CPUDispatcher):
        _get_nucleation_changes_per_state = numba.njit(
            _get_nucleation_changes_per_state)

    global _get_first_and_last_object_pos
    if not isinstance(_get_first_and_last_object_pos,
                      numba.core.registry.CPUDispatcher):
        _get_first_and_last_object_pos = numba.njit(
            _get_first_and_last_object_pos)

    global _get_tau
    if not isinstance(_get_tau,
                      numba.core.registry.CPUDispatcher):
        _get_tau = numba.njit(_get_tau)

    global _reset_local_density
    if not isinstance(_reset_local_density,
                      numba.core.registry.CPUDispatcher):
        _reset_local_density = numba.njit(
            _reset_local_density)

    global _get_local_and_total_density
    if not isinstance(_get_local_and_total_density,
                      numba.core.registry.CPUDispatcher):
        _get_local_and_total_density = numba.njit(
            _get_local_and_total_density)

    global _get_total_and_single_rates_for_state_transitions
    if not isinstance(_get_total_and_single_rates_for_state_transitions,
                      numba.core.registry.CPUDispatcher):
        _get_total_and_single_rates_for_state_transitions = numba.njit(
            _get_total_and_single_rates_for_state_transitions)


    global _get_property_action_changes
    if not isinstance(_get_property_action_changes,
                      numba.core.registry.CPUDispatcher):
        _get_property_action_changes = numba.njit(
            _get_property_action_changes)

    global _get_tmin_tmax_for_property_changes
    if not isinstance(_get_tmin_tmax_for_property_changes,
                      numba.core.registry.CPUDispatcher):
        _get_tmin_tmax_for_property_changes = numba.njit(
            _get_tmin_tmax_for_property_changes)

    global _get_times_of_next_transition
    if not isinstance(_get_times_of_next_transition,
                      numba.core.registry.CPUDispatcher):
        _get_times_of_next_transition = numba.njit(
            _get_times_of_next_transition)

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

    global _increase_lowest_no_object_idx
    if not isinstance(_increase_lowest_no_object_idx,
                      numba.core.registry.CPUDispatcher):
        _increase_lowest_no_object_idx = numba.njit(_increase_lowest_no_object_idx)

    global _reduce_highest_object_idx
    if not isinstance(_reduce_highest_object_idx,
                      numba.core.registry.CPUDispatcher):
        _reduce_highest_object_idx = numba.njit(_reduce_highest_object_idx)

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

    global _get_nucleation_on_objects_rate
    if not isinstance(_get_nucleation_on_objects_rate,
                      numba.cuda.dispatcher.CUDADispatcher):
        _get_nucleation_on_objects_rate = numba.cuda.jit(
            _get_nucleation_on_objects_rate, debug=debug, opt=opt)

    global _get_nucleation_changes_per_state
    if not isinstance(_get_nucleation_changes_per_state,
                      numba.cuda.dispatcher.CUDADispatcher):
        _get_nucleation_changes_per_state = numba.cuda.jit(
            _get_nucleation_changes_per_state, debug=debug, opt=opt)

    global _get_first_and_last_object_pos
    if not isinstance(_get_first_and_last_object_pos,
                      numba.cuda.dispatcher.CUDADispatcher):
        _get_first_and_last_object_pos = numba.cuda.jit(
            _get_first_and_last_object_pos, debug=debug, opt=opt)

    global _get_tau
    if not isinstance(_get_tau,
                      numba.cuda.dispatcher.CUDADispatcher):
        _get_tau = numba.cuda.jit(_get_tau, debug=debug, opt=opt)

    global _reset_local_density
    if not isinstance(_reset_local_density,
                      numba.cuda.dispatcher.CUDADispatcher):
        _reset_local_density = numba.cuda.jit(
        _reset_local_density, debug=debug, opt=opt)

    global _get_local_and_total_density
    if not isinstance(_get_local_and_total_density,
                      numba.cuda.dispatcher.CUDADispatcher):
        _get_local_and_total_density = numba.cuda.jit(
        _get_local_and_total_density, debug=debug, opt=opt)

    global _get_total_and_single_rates_for_state_transitions
    if not isinstance(_get_total_and_single_rates_for_state_transitions,
                      numba.cuda.dispatcher.CUDADispatcher):
        _get_total_and_single_rates_for_state_transitions = numba.cuda.jit(
        _get_total_and_single_rates_for_state_transitions, debug=debug, opt=opt)


    global _get_property_action_changes
    if not isinstance(_get_property_action_changes,
                      numba.cuda.dispatcher.CUDADispatcher):
        _get_property_action_changes = numba.cuda.jit(
        _get_property_action_changes, debug=debug, opt=opt)

    global _get_tmin_tmax_for_property_changes
    if not isinstance(_get_tmin_tmax_for_property_changes,
                      numba.cuda.dispatcher.CUDADispatcher):
        _get_tmin_tmax_for_property_changes = numba.cuda.jit(
            _get_tmin_tmax_for_property_changes, debug=debug,
            opt=opt)

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

    global _increase_lowest_no_object_idx
    if not isinstance(_increase_lowest_no_object_idx,
                      numba.cuda.dispatcher.CUDADispatcher):
        _increase_lowest_no_object_idx = numba.cuda.jit(_increase_lowest_no_object_idx,
                                                        debug=debug, opt=opt)

    global _reduce_highest_object_idx
    if not isinstance(_reduce_highest_object_idx,
                      numba.cuda.dispatcher.CUDADispatcher):
        _reduce_highest_object_idx = numba.cuda.jit(_reduce_highest_object_idx,
                                               debug=debug, opt=opt)

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


def _get_nucleation_on_objects_rate(some_creation_on_objects,
                                    transition_parameters,
                                    parameter_value_array, core_id, sim_id, param_id):
    # get nucleation on objects rate
    transition_nb = 0
    transition_nb_creation_on_objects = math.nan
    while transition_nb < some_creation_on_objects.shape[0]:
        if some_creation_on_objects[transition_nb] == 1:
            transition_nb_creation_on_objects = transition_nb
            break
        transition_nb += 1

    if math.isnan(transition_nb_creation_on_objects):
        return math.nan

    parameter_nb = transition_parameters[int(transition_nb_creation_on_objects)]
    nucleation_on_objects_rate = parameter_value_array[int(parameter_nb),
                                                       sim_id, param_id]

    return nucleation_on_objects_rate

def _get_property_action_changes(property_changes_per_state,
                                total_property_changes,
                                action_parameters, parameter_value_array,
                                action_state_array, all_action_properties,
                                action_operation_array,  core_id, sim_id, param_id):
    # Before iterating, check for each object the minimum and
    # maximum time it contributes to changing nucleation rate and
    # the direction for position property and non position property.
    # For that first get the net change for each property.
    # property_changes_per_state has following dimensions:
    # (states (including 0 state), properties)
    # with the state index being one lower than the state number
    action_nb = 0
    while action_nb < action_parameters.shape[0]:
        action_parameter = int(action_parameters[action_nb])
        action_value = parameter_value_array[action_parameter,
                                             sim_id, param_id]
        action_states = action_state_array[action_nb]
        action_property_nb = int(all_action_properties[action_nb, 0])
        action_operation = action_operation_array[action_nb]
        action_val_sign = action_operation * action_value
        if action_states[0] == 0:
            state_nb = 0
            while state_nb < property_changes_per_state.shape[0]:
                property_changes_per_state[state_nb,
                                           action_property_nb,
                                           sim_id,
                                           param_id] += action_val_sign
                total_property_changes[state_nb,
                                       sim_id, param_id] += action_val_sign
                state_nb += 1
        else:
            action_state_nb = 0
            val = action_val_sign
            while action_state_nb < action_states.shape[0]:
                if math.isnan(action_states[action_state_nb]):
                    break
                action_state = int(action_states[action_state_nb])
                property_changes_per_state[action_state-1,
                                           action_property_nb,
                                           sim_id, param_id] += val
                total_property_changes[action_state-1,
                                       sim_id, param_id] += val
                action_state_nb += 1
        action_nb += 1

    return property_changes_per_state, total_property_changes

def _get_nucleation_changes_per_state(nucleation_changes_per_state,
                                      property_changes_per_state,
                                      nucleation_on_objects_rate,
                                      core_id, sim_id, param_id):
    state_nb = 0
    while state_nb < property_changes_per_state.shape[0]:
        property_nb = 0
        while property_nb < property_changes_per_state.shape[1]:
            property_change = property_changes_per_state[state_nb, property_nb,
                                                         sim_id, param_id]
            nucleation_changes_per_state[state_nb,
                                        property_nb,
                                         sim_id,
                                         param_id] = (property_change *
                                                      nucleation_on_objects_rate)
            property_nb += 1
        state_nb += 1

def _get_first_and_last_object_pos(nb_objects, nb_parallel_cores, core_id):
        objects_per_core = (nb_objects / nb_parallel_cores)
        object_pos = objects_per_core * core_id
        last_object_pos = objects_per_core * (core_id + 1)
        last_object_pos = min(last_object_pos, nb_objects)
        return int(object_pos), int(last_object_pos)

def _reset_local_density(local_density, nb_parallel_cores, core_id,
                         sim_id, param_id):
    nb_x_pos = local_density.shape[0]
    (x_pos,
     last_x_pos) = _get_first_and_last_object_pos(nb_x_pos,
                                                  nb_parallel_cores,
                                                  core_id)
    while x_pos < last_x_pos:
        local_density[x_pos, sim_id, param_id] = 0
        x_pos += 1

def _get_local_and_total_density(local_density, total_density, local_resolution,
                                 highest_idx_with_object, properties_array,
                                 object_states, nb_parallel_cores, grid,
                                 thread_masks,
                                 core_id, sim_id, param_id):
        # get density of MTs at timepoint
        if core_id == 0:
            total_density[sim_id, param_id] = 0

        if nb_parallel_cores > 1:
            cuda.syncwarp(int(thread_masks[sim_id, param_id]))

        nb_objects = highest_idx_with_object[sim_id, param_id] + 1
        (object_pos,
         last_object_pos) = _get_first_and_last_object_pos(nb_objects,
                                                        nb_parallel_cores,
                                                        core_id)

        while object_pos < last_object_pos:
            if object_states[object_pos, sim_id, param_id] > 0:

                start = properties_array[0,object_pos,sim_id, param_id]
                end = start + properties_array[1,object_pos,sim_id, param_id]
                end += properties_array[2,object_pos,sim_id, param_id]
                if start != end:
                    x_start = int(math.floor(start / local_resolution))
                    x_end = int(math.ceil(end / local_resolution))
                    position_at_end = False
                    if x_end > local_density.shape[0]:
                        x_end = local_density.shape[0]
                        position_at_end = True
                    x_pos = max(x_start, 0)
                    while x_pos < x_end:
                        if (x_pos == (x_end - 1)) & (not position_at_end):
                            # for the x bin in which the MT ended, don't add a full MT
                            # but just the relative amount of the bin crossed by the MT
                            x_um = x_pos * local_resolution
                            diff = (end - x_um) / local_resolution
                            cuda.atomic.add(local_density,
                                            (x_pos, sim_id, param_id), diff)
                            cuda.atomic.add(total_density,
                                            (sim_id, param_id), diff)
                        else:
                            cuda.atomic.add(local_density,
                                            (x_pos, sim_id, param_id), 1)
                            cuda.atomic.add(total_density,
                                            (sim_id, param_id), 1)
                        x_pos += 1
            object_pos += 1


def _get_total_and_single_rates_for_state_transitions(parameter_value_array,
                                                      transition_parameters,
                                                      all_transition_states,
                                                      current_transition_rates,
                                                      total_rates,
                                                      nb_objects_all_states,
                                                      creation_on_objects,
                                                      total_density,
                                                      local_resolution,
                                                      nb_parallel_cores, grid,
                                                      thread_masks,
                                                      core_id, sim_id, param_id):

    if core_id == 0:
        total_rates[sim_id, param_id] = 0

    if nb_parallel_cores > 1:
        cuda.syncwarp(int(thread_masks[sim_id, param_id]))

    # get rates for all state transitions, depending on number of objects
    # in corresponding start state of transition
    nb_transitions = transition_parameters.shape[0]
    (transition_nb,
     last_transition_nb) = _get_first_and_last_object_pos(nb_transitions,
                                                          nb_parallel_cores,
                                                          core_id)

    while transition_nb < last_transition_nb:
        transition_states = all_transition_states[transition_nb]
        start_state = transition_states[0]
        transition_parameter = int(transition_parameters[transition_nb])
        transition_rate = parameter_value_array[transition_parameter,
                                                sim_id, param_id]
        if start_state == 0:
            # for state 0, the number of objects in state 0 does not matter
            if creation_on_objects[transition_nb] == 1:
                current_transition_rates[transition_nb,
                                         sim_id, param_id] = (transition_rate *
                                                              total_density[sim_id, param_id] *
                                                              local_resolution)
            else:
                current_transition_rates[transition_nb,
                                         sim_id, param_id] = transition_rate
        else:
            nb_objects = nb_objects_all_states[int(start_state-1), 
                                               sim_id, param_id]
            current_transition_rates[transition_nb,
                                     sim_id, param_id] = (nb_objects *
                                                          transition_rate)
        current_rate = current_transition_rates[transition_nb, sim_id, param_id]

        cuda.atomic.add(total_rates, (sim_id, param_id), current_rate)

        transition_nb += 1



def _get_tmin_tmax_for_property_changes(property_changes_tmax_array,
                                       property_changes_tmin_array,
                                       property_changes_per_state,
                                       total_property_changes,
                                       current_sum_tmax,
                                       properties_tmax_sorted,
                                       properties_tmax,
                                       properties_tmax_array,
                                       end_position_tmax_array,

                                       highest_idx_with_object,
                                       object_states, properties_array,
                                       property_max_values,
                                        nb_parallel_cores, grid, core_id,
                                       sim_id, param_id
                                       ):

    nb_objects = highest_idx_with_object[sim_id, param_id] + 1
    (object_nb,
     last_object_pos) = _get_first_and_last_object_pos(nb_objects,
                                                    nb_parallel_cores,
                                                    core_id)
    while object_nb < last_object_pos:
        # check if action should be executed on object
        state = int(object_states[object_nb, sim_id, param_id])
        if state > 0:
            properties_net_change = property_changes_per_state[state-1,:,
                                                               sim_id,
                                                               param_id]
            # get all tmax for properties that are reduced
            # which is the maximum time that properties can be reduced
            # before they are 0
            # ignore the position property (idx 0)
            net_change_property_nb = 1
            while net_change_property_nb < properties_net_change.shape[0]:
                net_change = properties_net_change[net_change_property_nb]
                if net_change < 0:
                    tmax = - (properties_array[net_change_property_nb,
                                            object_nb,
                                            sim_id, param_id] / net_change)

                    properties_tmax[core_id, net_change_property_nb-1,
                                    sim_id, param_id] = tmax
                net_change_property_nb += 1
            # sort indices of properties by size of tmax (smaller to larger)

            nb_sorted_tmax = 0
            if not math.isnan(properties_tmax[core_id, 0, sim_id, param_id]):
                properties_tmax_sorted[core_id, 0, sim_id, param_id] = 0
                nb_sorted_tmax += 1

            property_idx = 1
            while property_idx < properties_tmax.shape[1]:
                tmax = properties_tmax[core_id, property_idx, sim_id, param_id]
                if not math.isnan(tmax):
                    nb_sorted_tmax += 1
                    # go through already sorted tmax indices and check which idx
                    # is current tmax smaller than
                    property_idx_inner = 0
                    while property_idx_inner < nb_sorted_tmax:
                        tmax_inner = properties_tmax_sorted[core_id,
                                                            property_idx_inner,
                                                            sim_id,
                                                            param_id]
                        # if the index which the current tmax is smaller than
                        # is found, move the following sorted indices
                        # one index down, starting with the highest idx
                        # to prevent overwriting
                        if tmax < tmax_inner:
                            property_idx_new_sort = property_idx
                            while property_idx_new_sort > property_idx_inner:
                                properties_tmax_sorted[core_id,
                                                       property_idx_new_sort,
                                                       sim_id,
                                                       param_id] = properties_tmax_sorted[core_id,
                                                                                          property_idx_new_sort-1,
                                                                                          sim_id, param_id]
                                property_idx_new_sort -= 1
                            properties_tmax_sorted[core_id,
                                                   property_idx_inner,
                                                   sim_id,
                                                   param_id] = property_idx
                            break
                        # if it was not smaller than any previously sorted tmax
                        # add it at the end
                        if property_idx_inner == property_idx:
                            properties_tmax_sorted[core_id,
                                                   property_idx,
                                                   sim_id,
                                                   param_id] = property_idx
                        property_idx_inner += 1
                property_idx += 1

            # simulate changes of length and position over time
            # by separating it into sections in which sets of actions
            # are active
            # Do this by going through the maximum times that each property
            # is changed.
            # save intermediate results for position, all properties,
            # the end position, the time and the current total sum of speeds
            # dimensions of properties_tmax_array are (properties, properties)

            current_sum_tmax[core_id, 0, sim_id,
                             param_id] = total_property_changes[state-1, sim_id,
                                                                param_id]

            property_tmax_nb = 0
            while property_tmax_nb < properties_tmax_array.shape[2]:
                # one has to be added to the property index since the
                # position was not considered
                property_val = properties_array[property_tmax_nb,
                                                object_nb,
                                                sim_id, param_id]
                properties_tmax_array[core_id, 0, property_tmax_nb,
                                      sim_id, param_id] = property_val
                property_tmax_nb += 1

            first_position = properties_array[0, object_nb,
                                              sim_id, param_id]

            tmax_end_position = math.nan
            tmax_start_position = math.nan
            if first_position >= 0:
                tmax_start_position = 0
            tmax_object_removal = math.nan

            tmax_idx = 0
            last_tmax = 0
            while tmax_idx < nb_sorted_tmax:
                if tmax_idx == 0:
                    last_idx = 0
                else:
                    last_idx = tmax_idx - 1
                property_idx = int(properties_tmax_sorted[core_id,
                                                          tmax_idx,
                                                          sim_id, param_id])
                tmax = properties_tmax[core_id, property_idx, sim_id, param_id]
                # update lengths and positions
                delta_t = tmax - last_tmax
                property_net_change_nb = 0
                end_position = 0
                while property_net_change_nb < properties_net_change.shape[0]:
                    net_change = properties_net_change[property_net_change_nb]
                    old_val = properties_tmax_array[core_id,
                                                    last_idx,
                                                    property_net_change_nb,
                                                    sim_id, param_id]

                    if old_val == 0:
                        new_val = old_val
                    else:
                        if net_change == 0:
                            new_val = old_val
                        else:
                            new_val = (old_val + net_change * delta_t)
                    # this seems to have been the code block that caused
                    # the sync error. But uncommenting the commented code below
                    # until tmax_idx += 1 also lead to a sync errors
                    properties_tmax_array[core_id, tmax_idx,
                                          property_net_change_nb,
                                          sim_id, param_id] = new_val
                    end_position = end_position + new_val
                    property_net_change_nb += 1
                end_position_tmax_array[core_id, tmax_idx,
                                        sim_id, param_id] = end_position
                # if the end positions goes below zero, the object is removed
                # (technically only for 1D compartments and if the there is
                #   and object removal condition)
                if end_position < 0:
                    # get the time of object removal as
                    # the difference of the end_position to the max position
                    # divided by the total net change
                    position_diff = end_position - property_max_values[0,0]
                    tmax_object_removal = (position_diff /
                                           current_sum_tmax[core_id,
                                                            tmax_idx,
                                                            sim_id,
                                                            param_id])
                    break

                # get time at which end_position reaches the max position
                # don't allow the tmax_end_position to be overwritten once
                # defined
                if ((end_position > property_max_values[0,0]) &
                        (math.isnan(tmax_end_position))):
                    if current_sum_tmax[core_id,tmax_idx, sim_id, param_id] > 0:
                        # the difference from the current tmax is
                        # the difference of the end_position minus the
                        # max position, divided by the current sum
                        position_diff = end_position - property_max_values[0,0]
                        tmax_diff = position_diff / current_sum_tmax[core_id,
                                                                     tmax_idx,
                                                                     sim_id,
                                                                     param_id]
                        tmax_end_position = tmax - tmax_diff

                # get time at which position is completely within
                # compartment
                # don't allow the tmax_start_position to be overwritten once
                # defined
                if ((first_position < 0) &
                        (math.isnan(tmax_start_position))):
                    if properties_tmax_array[core_id,
                                             tmax_idx, 0,
                                             sim_id, param_id] > 0:
                        if properties_net_change[0] > 0:
                            # the difference from the current tmax is the
                            # position divided by the net_change of the
                            position = properties_tmax_array[core_id,
                                                             tmax_idx, 0,
                                                             sim_id,
                                                             param_id]
                            tmax_start_position = (position /
                                                   properties_net_change[0])

                last_tmax = tmax
                # for next section, remove net changes from sum of all net
                # changes that
                current_sum_tmax[core_id,
                                 tmax_idx+1,
                                 sim_id,
                                 param_id] = (current_sum_tmax[core_id,
                                                               tmax_idx,
                                                               sim_id,
                                                               param_id] -
                                              properties_net_change[property_idx])
                tmax_idx += 1

            # if the object has not been removed, make sure that all end
            # points have been calculated already, otherwise calculate them
            # now
            if not math.isnan(tmax_object_removal):
                if math.isnan(tmax_end_position):
                    # calculate tmax_end_position
                    if current_sum_tmax[core_id,tmax_idx, sim_id, param_id] > 0:
                        position_diff = (property_max_values[0,0] -
                                         end_position)
                        tmax_end_position = (position_diff /
                                             current_sum_tmax[core_id,
                                                              tmax_idx,
                                                              sim_id,
                                                              param_id])
                if (math.isnan(tmax_start_position)):
                    # calculate tmax-start_position
                    if properties_net_change[0] > 0:
                        position = properties_tmax_array[core_id,
                                                         tmax_idx - 1, 0,
                                                         sim_id, param_id]
                        tmax_start_position = (position /
                                               properties_net_change[0])

            # At this point there is an array that saved the state of
            # the microtubule (length and position) for each property with
            # a negative net change after this property has become 0
            #
            #
            # for each net change (positive or negative) of position or
            # another property there is a condition after which the net
            # change cannot be executed any longer.
            # For all changes, end position < 0 will be the latest possible
            # end point.
            # for reducing the position:
            #   end_position == max & total_net_change > 0
            # for increasing the position:
            #   position >= 0 | ((end_position == max) & (total_net_change > 0))
            # for reducing another property:
            #   No additional condition
            # for increasing another property:
            #   (end_position == max) & (total_net_change > 0)

            net_change_property_nb = 0
            while net_change_property_nb < properties_net_change.shape[0]:
                net_change = properties_net_change[net_change_property_nb]
                if net_change != 0:
                    tmax = math.nan
                    tmin = 0
                    if not math.isnan(tmax_object_removal):
                        tmax = tmax_object_removal
                    if net_change_property_nb == 0:
                        if net_change < 0:
                            position = properties_array[0, object_nb,
                                                        sim_id, param_id]
                            if position <= 0:
                                tmin = 0
                            else:
                                tmin = - (position / net_change)

                            # for reducing the position:
                            # end_position == max & total_net_change > 0
                            if not math.isnan(tmax_end_position):
                                if math.isnan(tmax):
                                    tmax = tmax_end_position
                                else:
                                    tmax = min(tmax, tmax_end_position)
                        else:
                            # for increasing the position:
                            # position >= 0 | ((end_position == max) &
                            #                   (total_net_change > 0))
                            if not math.isnan(tmax_start_position):
                                if math.isnan(tmax):
                                    tmax = tmax_start_position
                                else:
                                    tmax = min(tmax, tmax_start_position)

                    else:
                        properties_idx = net_change_property_nb
                        property_tmax = properties_tmax[core_id,
                                                        properties_idx - 1,
                                                        sim_id, param_id]
                        if net_change < 0:
                            # only consider object removal tmax additionally
                            # (which is also considered for all other
                            #  conditions)
                            if not math.isnan(property_tmax):
                                if math.isnan(tmax):
                                    tmax = property_tmax
                                else:
                                    tmax = min(tmax, property_tmax)
                        else:
                            # for increasing another property:
                            #   (end_position == max) & (total_net_change > 0)
                            if not math.isnan(tmax_end_position):
                                if math.isnan(tmax):
                                    tmax = tmax_end_position
                                else:
                                    tmax = min(tmax, tmax_end_position)

                    property_changes_tmin_array[object_nb,
                                                net_change_property_nb,
                                                sim_id, param_id] = tmin

                    property_changes_tmax_array[object_nb,
                                                net_change_property_nb,
                                                sim_id, param_id] = tmax

                net_change_property_nb += 1
        object_nb += 1


def _get_tau(total_rates, highest_idx_with_object, object_states,
            property_changes_per_state,nucleation_changes_per_state,
            property_changes_tmin_array,property_changes_tmax_array,
             constant, second_order,
             rng_states, simulation_factor, parameter_factor,
             nb_parallel_cores, grid, thread_masks, core_id, sim_id, param_id):

        rate_baseline = total_rates[sim_id, param_id]
        random_nb = _get_random_number(sim_id, param_id, rng_states,
                                       simulation_factor, parameter_factor)
        rand_exp = - (math.log(random_nb))

        tau_guess = rand_exp/rate_baseline
        first_order = rate_baseline

        second_order[0, sim_id, param_id] = 0
        constant[0, sim_id, param_id] = - rand_exp

        tau_error = 1
        lowest_error = 1
        best_tau = 0
        nb = 0
        while tau_error > 0.005:
            second_order[1, sim_id, param_id] = 0
            constant[1, sim_id, param_id] = 0

            if nb == 0:
                calculate_second_order_base = True
            else:
                calculate_second_order_base = False

            nb_objects = highest_idx_with_object[sim_id, param_id] + 1
            (object_nb,
             last_object_nb) = _get_first_and_last_object_pos(nb_objects,
                                                              nb_parallel_cores,
                                                              core_id)
            while object_nb < last_object_nb:
                state = object_states[object_nb, sim_id, param_id]
                if state > 0:
                    property_idx = 0
                    while property_idx < property_changes_per_state.shape[1]:
                        net_change = nucleation_changes_per_state[state-1,
                                                                property_idx,
                                                                sim_id,
                                                                param_id]
                        if net_change != 0:
                            tmin = property_changes_tmin_array[object_nb,
                                                               property_idx,
                                                               sim_id, param_id]
                            tmax = property_changes_tmax_array[object_nb,
                                                               property_idx,
                                                               sim_id, param_id]
                            # if tmin is nan then this property change does not
                            # lead to a change in density
                            if not math.isnan(tmin):
                                if tau_guess > tmin:
                                    # if tmax is nan this property change leads to a
                                    # change in density without time limit
                                    if math.isnan(tmax) & calculate_second_order_base:
                                        # add net_change to variable for tau dependent
                                        # change, also add to base level for that
                                        # variable, so that this does not need to
                                        # be added again (second_order)
                                        cuda.atomic.add(second_order,
                                                        (0, sim_id, param_id),
                                                        net_change)
                                        cuda.atomic.add(constant,
                                                        (0, sim_id, param_id),
                                                        - (tmin**2)/2)
                                    elif tmax > 0:
                                        if tau_guess < tmax:
                                            # if tau is before tmax, add net_change
                                            # to variable for tau dependent change
                                            # (second_order)
                                            cuda.atomic.add(second_order,
                                                            (1, sim_id, param_id),
                                                            net_change)
                                            cuda.atomic.add(constant,
                                                            (1, sim_id, param_id),
                                                            - net_change *
                                                            (tmin**2)/2)

                                        else:
                                            # if tau is smaller then tmax, add
                                            # net_change to variable for tau independent
                                            # change (constant)
                                            cuda.atomic.add(constant,
                                                            (1, sim_id, param_id),
                                                            (net_change *
                                                             (tmax**2 -
                                                              tmin**2) / 2))

                        property_idx += 1
                object_nb += 1

            if nb_parallel_cores > 1:
               cuda.syncwarp(int(thread_masks[sim_id, param_id]))

            total_constant = (constant[0, sim_id, param_id] +
                              constant[1, sim_id, param_id])
            # divide by two due to integral
            total_second_order = (second_order[0, sim_id, param_id] +
                                  second_order[1, sim_id, param_id])/2

            if total_second_order != 0:
                new_tau_guess_sqrt = math.sqrt(first_order**2 -
                                               4*total_second_order*
                                               total_constant)
                new_tau_guess_min = - first_order - new_tau_guess_sqrt
                new_tau_guess_max = - first_order + new_tau_guess_sqrt
                if new_tau_guess_min < 0:
                    new_tau_guess = new_tau_guess_max
                else:
                    new_tau_guess = new_tau_guess_min

                new_tau_guess = new_tau_guess / (2 * total_second_order)
            else:
                new_tau_guess = - total_constant/first_order

            # check whether error is small enough to stop
            tau_error = abs(tau_guess - new_tau_guess) / tau_guess

            if tau_error < lowest_error:
                lowest_error = tau_error
                best_tau = new_tau_guess

            # stop trying to get ideal tau after some iterations
            # then take the best tau (lowest error) until that point
            if nb == 5:
                tau_guess = best_tau
                break

            tau_guess = tau_guess + (new_tau_guess - tau_guess) * 0.5
            nb += 1

        return tau_guess


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
                               current_transitions,
                               sim_id, param_id,
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
                                        object_states, core_id,
                                        sim_id, param_id, rng_states,
                                        highest_idx_with_object,
                                        lowest_idx_no_object,
                                        simulation_factor, parameter_factor):

    # the transitions masks only tell which reaction happens in each
    # stimulation, but not which object in this simulation is affected
    # To get one random object of the possible objects,
    # first, create mask of index positions, so that each object for each
    # simulation has a unique identifier (index) within this simulation
    # setting all positions where no catastrophe can take place to 0
    transition_nb = current_transitions[sim_id, param_id]
    all_transition_positions[sim_id, param_id] = math.nan
    if math.isnan(transition_nb):
        return

    start_state = all_transition_states[int(transition_nb), 0]
    # if start state is 0, choose the first object with state 0
    if start_state == 0:
        # if the possible lowest idx with no object actually has an object now
        # increase index until an index without object is found and save this
        # position
        if core_id == 0:
            if object_states[lowest_idx_no_object[sim_id, param_id],
                             sim_id, param_id] != 0:
                _increase_lowest_no_object_idx(lowest_idx_no_object,
                                               object_states, sim_id, param_id)
            all_transition_positions[sim_id,
                                     param_id] = lowest_idx_no_object[sim_id,
                                                                      param_id]
        return
    else:
        # for all other states, choose a random position with that state
        nb_objects = nb_objects_all_states[int(start_state-1), sim_id, param_id]
        random_object_pos = math.floor(_get_random_number(sim_id, param_id,
                                                     rng_states,
                                                     simulation_factor,
                                                     parameter_factor)
                                       * nb_objects)
        # allow a maximum of nb_objects - 1
        random_object_pos = int(min(nb_objects-1, random_object_pos))
        object_pos = 0
        current_nb_state_objects = 0

        # if the highest idx with object vanished
        # search for the new highest object idx
        if object_states[highest_idx_with_object[sim_id, param_id],
                         sim_id, param_id] == 0:
            _reduce_highest_object_idx(highest_idx_with_object,
                                       object_states, sim_id, param_id)

        # go through all objects, check which one is in the start_state
        # and then choose the nth (n=random_object_pos) object that is in
        # the start_state
        while object_pos <= highest_idx_with_object[sim_id, param_id]:# object_states.shape[0]:#
            object_state = object_states[object_pos, sim_id, param_id]
            # if object_state > 4:
            #     print(555, object_state, object_pos)
            if object_state == start_state:
                if current_nb_state_objects == random_object_pos:
                    all_transition_positions[sim_id, param_id] = object_pos
                    return
                current_nb_state_objects += 1
            object_pos += 1

    return


def _execute_actions_on_objects(parameter_value_array, action_parameters,
                                action_state_array,
                                properties_array,
                                property_min_values,
                                property_max_values,
                                all_action_properties,
                                action_operation_array,
                                object_states,
                                reaction_time,
                                highest_idx_with_object,
                                nb_parallel_cores, grid, core_id,
                                sim_id, param_id):
    # execute actions on objects depending on state, before changing state
    action_nb = 0

    while action_nb < action_parameters.shape[0]:
        action_parameter = int(action_parameters[action_nb])
        # get the current action value, dependent on reaction time
        action_value = parameter_value_array[action_parameter, sim_id, param_id]
        if action_value != 0:
            current_action_value = action_value * reaction_time
            action_states = action_state_array[action_nb]
            action_properties = all_action_properties[action_nb]
            # action operation is -1 for subtracting and 1 for adding
            action_operation = action_operation_array[action_nb]
            # go through each object and check whether its state is in
            # action_states,

            nb_objects = highest_idx_with_object[sim_id, param_id] + 1
            (object_pos,
             last_object_pos) = _get_first_and_last_object_pos(nb_objects,
                                                            nb_parallel_cores,
                                                            core_id)
            while object_pos < last_object_pos:
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
                            changed_start_values_array,
                            creation_on_objects,
                          highest_idx_with_object,lowest_idx_no_object,

                          local_density, total_density, local_resolution,
                          nb_parallel_cores, grid, core_id,
                          sim_id, param_id,
                          rng_states, simulation_factor, parameter_factor):

    # update the simulations according to executed transitions
    transition_number = current_transitions[sim_id, param_id]
    if not math.isnan(transition_number):
        transition_number = int(transition_number)
        transition_states = all_transition_states[transition_number]
        start_state = transition_states[0]
        end_state = transition_states[1]
        transition_position = all_transition_positions[sim_id, param_id]
        object_states[int(transition_position),sim_id, param_id] = end_state
        # change the object counter according to the executed transition
        if core_id == 0:
            if start_state != 0:
                nb_objects_all_states[int(start_state)-1, sim_id, param_id] -= 1
            if end_state != 0:
                nb_objects_all_states[int(end_state)-1, sim_id, param_id] += 1

        # change property values based on transitions
        if start_state == 0:
            # check if the idx for creating a new object is higher than
            # the currently highest idx
            if transition_position > highest_idx_with_object[sim_id, param_id]:
                highest_idx_with_object[sim_id, param_id] = transition_position

            # if a new object was created, set property values according to
            # defined value
            nb_properties = property_start_vals.shape[0]
            (property_nb,
             last_property_nb) = _get_first_and_last_object_pos(nb_properties,
                                                              nb_parallel_cores,
                                                              core_id)
            while property_nb < last_property_nb:

                if ((creation_on_objects[transition_number] == 1) &
                        (property_nb == 0)):
                    random_nb = _get_random_number(sim_id, param_id, rng_states,
                                                   simulation_factor,
                                                   parameter_factor)
                    # go through local density until threshold is reached
                    x_pos = 0
                    threshold = random_nb * total_density[sim_id, param_id]
                    density_sum = 0
                    while x_pos < local_density.shape[0]:
                        local_density_here = local_density[x_pos, sim_id, param_id]
                        density_sum = density_sum + local_density_here
                        if density_sum > threshold:
                            property_val = (x_pos + 1) * local_resolution
                            # assume linear transition of density
                            # from last point to this point
                            # the exact point of crossing the threshold can then
                            # easily be determined as that much below the current x
                            # as the fraction of the added density at the current
                            # x position that the sum is above the threshold
                            # e.g. if the sum is almost the entirety of the density
                            # added at the current x position above the threshold
                            # then the actual x position at which the threshold is
                            # crossed should be very close to the last x position
                            property_val -= ((density_sum - threshold) /
                                             local_density_here) * local_resolution
                            break
                        x_pos += 1
                else:
                    # check whether the property start values are different for the
                    # current transition
                    if math.isnan(changed_start_values_array[transition_number,
                                                             property_nb, 0]):
                        property_start_val = property_start_vals[property_nb]
                    else:
                        property_start_val = changed_start_values_array[transition_number,
                                                                        property_nb]
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
            # check if the idx for removing a position is the currently highest
            # position
            # if so, go backward from that position until the next object is found
            if core_id == 0:
                if transition_position < lowest_idx_no_object[sim_id, param_id]:
                    lowest_idx_no_object[sim_id, param_id] = transition_position

            # if an object was removed, set property values to NaN
            nb_properties = property_start_vals.shape[0]
            (property_nb,
             last_property_nb) = _get_first_and_last_object_pos(nb_properties,
                                                              nb_parallel_cores,
                                                              core_id)
            while property_nb < last_property_nb:
                properties_array[property_nb,
                                 int(transition_position),
                                 sim_id, param_id] = math.nan
                property_nb += 1

        # if the object is not removed, it might be that the transition triggers
        # other events as well
        if (end_state != 0) & (core_id == 0):
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
                                 sim_id, param_id] = source_val + target_val

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

def _increase_lowest_no_object_idx(lowest_idx_no_object,
                                   object_states, sim_id, param_id):
    # if so, go forward from that position until the next empty idx is found
    object_pos = int(lowest_idx_no_object[sim_id, param_id])
    while object_pos < object_states.shape[0]:
        if object_states[object_pos, sim_id, param_id] == 0:
            break
        object_pos = object_pos + 1
    lowest_idx_no_object[sim_id, param_id] = object_pos

def _reduce_highest_object_idx(highest_idx_with_object,
                               object_states, sim_id, param_id):
    # if so, go backward from that position until the next object is found
    object_pos = int(highest_idx_with_object[sim_id, param_id])
    while object_pos > 0:
        if object_states[object_pos, sim_id, param_id] > 0:
            break
        object_pos = object_pos - 1
    highest_idx_with_object[sim_id, param_id] = object_pos

def _remove_objects(all_object_removal_properties, object_removal_operations,
                    nb_objects_all_states, object_states, properties_array,
                    highest_idx_with_object, lowest_idx_no_object,
                    nb_parallel_cores, grid, core_id, sim_id, param_id):
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

        nb_objects = highest_idx_with_object[sim_id, param_id] + 1
        (object_pos,
         last_object_pos) = _get_first_and_last_object_pos(nb_objects,
                                                          nb_parallel_cores,
                                                          core_id)
        while object_pos < last_object_pos:
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

                    cuda.atomic.min(lowest_idx_no_object,
                                    (sim_id, param_id), object_pos)
                    # if object_pos < lowest_idx_no_object[sim_id, param_id]:
                    #     lowest_idx_no_object[sim_id,
                    #                          param_id] = object_pos

                    object_state_to_remove = object_states[int(object_pos),
                                                           sim_id, param_id]

                    cuda.atomic.add(nb_objects_all_states,
                                    (object_state_to_remove-1,sim_id, param_id),
                                    -1)

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
                                          highest_idx_with_object,
                                          nb_parallel_cores, grid, thread_masks,
                                          core_id, sim_id, param_id):

    if core_id == 0:
        current_timepoint = current_timepoint_array[sim_id, param_id]
        # check .if the next timepoint was reached, then save all values
        # at correct position
        time_jump = math.floor((times[0,sim_id, param_id] -
                                     (current_timepoint * time_resolution[0]))
                                    / time_resolution[0])
        current_timepoint += time_jump

        current_timepoint_array[sim_id, param_id] = current_timepoint

    if nb_parallel_cores > 1:
        cuda.syncwarp(int(thread_masks[sim_id, param_id]))

    current_timepoint = current_timepoint_array[sim_id, param_id]
    # if the initial state was not saved then the index is actually the
    # timepoint minus 1
    if save_initial_state:
        timepoint_idx = int(current_timepoint)
    else:
        timepoint_idx = int(current_timepoint) - 1

    # make sure that timepoint_idx is not higher than timepoint_array shape
    timepoint_idx = min(timepoint_idx, timepoint_array.shape[0]-1)

    if core_id == 0:
        timepoint_array[timepoint_idx, sim_id, param_id] = current_timepoint

    # copy all current data into time-resolved data
    nb_objects = highest_idx_with_object[sim_id, param_id] + 1
    (object_pos,
     last_object_pos) = _get_first_and_last_object_pos(nb_objects,
                                                      nb_parallel_cores,
                                                      core_id)
    while object_pos < last_object_pos:
        if object_pos > 0:
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

    # no synchronization needed since no downstream processing
    # needs these synchronized changes, except the end of all simulations
    # which already include a synchronization step