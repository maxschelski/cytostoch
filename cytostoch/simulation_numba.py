import numpy as np

# NUMBA_DEBUG=1
# NUMBA_DEVELOPER_MODE = 1
# NUMBA_DEBUGINFO = 1
import numba
import numba.cuda.random
from numba import int32, float32
import math
import cmath
from numba import cuda
import inspect

from matplotlib import pyplot as plt


def _execute_simulation_gpu(object_states, properties_array, times,
                            nb_simulations, nb_parameter_combinations,
                            parameter_value_array,
                            params_prop_dependence,
                            position_dependence,
                            object_dependent_rates,
                            transition_parameters,
                            all_transition_states,
                            action_parameters,
                            action_state_array,
                            all_action_properties,
                            action_operation_array,
                            current_transition_rates,
                            property_start_values,
                            property_extreme_values,
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
                            properties_tmax,
                            current_sum_tmax,

                            property_changes_tminmax_array,
                            property_changes_per_state,
                            nucleation_changes_per_state,
                            total_property_changes,
                            density_threshold_boundaries,

                            tau_square_eq_terms,

                            first_last_idx_with_object,
                                     local_object_lifetime_array,
                            local_lifetime_resolution,

                            timepoint_array,
                            time_resolution, min_time, start_save_time,
                            save_initial_state,

                            start_nb_parallel_cores, nb_parallel_cores,
                            thread_masks, thread_to_sim_id,

                            local_density, local_resolution, total_density,
                            rng_states, simulation_factor, parameter_factor
                        ):
    # np.random.seed(seed)
    iteration_nb = 0

    parameter_value_array = cuda.const.array_like(parameter_value_array)
    params_prop_dependence = cuda.const.array_like(params_prop_dependence)
    transition_parameters = cuda.const.array_like(transition_parameters)
    all_transition_states = cuda.const.array_like(all_transition_states)
    action_parameters = cuda.const.array_like(action_parameters)
    action_state_array = cuda.const.array_like(action_state_array)
    all_action_properties = cuda.const.array_like(all_action_properties)
    action_operation_array = cuda.const.array_like(action_operation_array)
    property_start_values = cuda.const.array_like(property_start_values)
    property_extreme_values = cuda.const.array_like(property_extreme_values)
    all_transition_tranferred_vals = cuda.const.array_like(all_transition_tranferred_vals)
    all_transition_set_to_zero_properties = cuda.const.array_like(all_transition_set_to_zero_properties)
    changed_start_values_array = cuda.const.array_like(changed_start_values_array)
    creation_on_objects = cuda.const.array_like(creation_on_objects)
    all_object_removal_properties = cuda.const.array_like(all_object_removal_properties)
    object_removal_operations = cuda.const.array_like(object_removal_operations)

    # local_density = cuda.shared.array((1,201,1,1), numba.float32)

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

    # reassign_threads_time_step_size = 0.1
    # min_time_diff_for_reassigning = 0.1
    # nb_reassigned_threads = 1
    # rel_time_reassignment_check = 0.1

    total_nb_simulations = (nb_simulations * nb_parameter_combinations *
                            start_nb_parallel_cores)
    # print(cuda.gridsize(1), total_nb_simulations)
    current_sim_nb = thread_id
    new_simulation = False
    # new_assignment = 0
    # re_assigned = False
    while current_sim_nb < total_nb_simulations:

        # For each parameter combination the defined number of simulations
        # are done on a defined number of cores
        if not new_simulation:
            param_id = int(math.floor(current_sim_nb /
                                      (nb_simulations * start_nb_parallel_cores)))
            sim_id = int(math.floor((current_sim_nb -
                                     param_id * nb_simulations * start_nb_parallel_cores)
                                    / start_nb_parallel_cores))
            core_id = int(current_sim_nb -
                          param_id * nb_simulations * start_nb_parallel_cores -
                          sim_id * start_nb_parallel_cores)

            thread_to_sim_id[thread_id, 0] = sim_id
            thread_to_sim_id[thread_id, 1] = param_id
            thread_to_sim_id[thread_id, 2] = core_id

            # warp_nb = thread_id // cuda.warpsize

            warp_thread_idx = cuda.laneid
            cuda.atomic.add(thread_masks, (0, sim_id, param_id),
                            (1 << warp_thread_idx))

            cuda.atomic.add(thread_masks, (1, sim_id, param_id),
                            (1 << warp_thread_idx))

            cuda.atomic.add(thread_masks, (2, sim_id, param_id),
                            (1 << warp_thread_idx))

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

            if nb_parallel_cores[sim_id, param_id] > 1:
                cuda.syncwarp(thread_masks[0,sim_id, param_id])

        while True:
            # # wait for old and new threads to be synced here
            # cuda.syncwarp(thread_masks[0, sim_id, param_id])

            # new threads should wait until the old threads of the simulation
            # have concluded the current iteration
            # syncwarp seems to require that no other threads in the mask
            # execute a syncwarp with another mask, therefore the last iteration
            # must be fully concluded before syncwarp can be triggered in the
            # new threads.
            if new_simulation:
                trial = 0
                while True:
                    # print(6666, thread_id, trial, thread_masks[0, sim_id, param_id])
                    if ((thread_masks[0, sim_id, param_id] == 0) |
                            (timepoint_array[0, sim_id, param_id] >=
                             math.floor(min_time / time_resolution[0]))):
                        break
                    # if trial == 10:
                    #     break
                    trial += 1

            # For newly assigned threads, check whether the simulation the
            # thread was assigned to already finished. If so, assign another
            # simulation
            if (timepoint_array[0, sim_id, param_id] >=
                    math.floor(min_time / time_resolution[0])):
                thread_masks[0, sim_id, param_id] = int(thread_masks[1, sim_id, param_id])
                thread_masks[2, sim_id, param_id] = int(thread_masks[1, sim_id, param_id])
                if new_simulation:
                    cuda.atomic.add(nb_parallel_cores, (sim_id, param_id), 1)
                    new_simulation = False
            else:
                    # print(thread_id, warp_nb, warp_thread_idx, iteration_nb,
                    #       sim_id, param_id)

                if thread_masks[1, sim_id, param_id] != thread_masks[2, sim_id, param_id]:
                    cuda.syncwarp(thread_masks[1, sim_id, param_id])
                    # if not new_simulation:
                    #     cuda.syncwarp(thread_masks[0, sim_id, param_id])
                    # thread_masks[0, sim_id, param_id] = int(thread_masks[1,sim_id,
                    #                                                      param_id])
                    if new_simulation:
                        new_sim_nb = 1
                    else:
                        new_sim_nb = 0
                    warp_nb = thread_id // cuda.warpsize
                    # print(999, thread_id, new_sim_nb, sim_id, core_id)
                    # print(999, thread_id, sim_id, param_id,
                    #       timepoint_array[0, sim_id, param_id]
                    #       )
                    # thread_masks[1, sim_id, param_id] = 0
                    # current_sim_nb += nb_processes
                    # break

                thread_masks[0, sim_id, param_id] = int(thread_masks[1, sim_id, param_id])
                thread_masks[2, sim_id, param_id] = int(thread_masks[1, sim_id, param_id])
                # thread_masks[0, sim_id, param_id] = int(thread_masks[2, sim_id, param_id])
                # cuda.syncwarp(thread_masks[0, sim_id, param_id])

                if new_simulation:
                    cuda.atomic.add(nb_parallel_cores, (sim_id, param_id), 1)
                    new_simulation = False

                # if thread_masks[0, sim_id, param_id] == 0:
                #     print(666, thread_id, thread_masks[0, sim_id, param_id])

                if new_simulation:
                    # print(thread_masks[1, sim_id, param_id], thread_id, core_id,
                    #       nb_parallel_cores[sim_id, param_id])
                    # new_simulation = False
                    success = 0
                    break

                success = _run_iteration(object_states,
                                         properties_array,
                                       times, parameter_value_array,
                                         params_prop_dependence,
                                         position_dependence,
                                        object_dependent_rates,
                                       transition_parameters, all_transition_states,
                                       action_parameters, action_state_array,
                                       all_action_properties, action_operation_array,
                                       current_transition_rates,
                                       property_start_values,
                                       property_extreme_values,
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
                                         properties_tmax,
                                         current_sum_tmax,

                                         property_changes_tminmax_array,
                                         property_changes_per_state,
                                        nucleation_changes_per_state,
                                         total_property_changes,
                                        density_threshold_boundaries,

                                         tau_square_eq_terms,

                                        first_last_idx_with_object,
                                     local_object_lifetime_array,
                            local_lifetime_resolution,

                                       timepoint_array, min_time,
                                         start_save_time,
                                       time_resolution, save_initial_state,

                                         nb_parallel_cores, thread_masks,

                                        local_density, local_resolution,total_density,

                                       rng_states, core_id, sim_id, param_id,
                                         simulation_factor, parameter_factor,
                               )

            # while the following code seems to work, it does not speed
            # up simulations - instead for some reason it slows it down?!

            if (timepoint_array[0, sim_id, param_id] >=
                    math.floor(min_time/time_resolution[0])):
                # print(thread_id, warp_nb, warp_thread_idx, iteration_nb,
                #       sim_id, param_id)
                cuda.syncwarp(thread_masks[1, sim_id, param_id])
                multi_core_opt = False
                current_sim_nb += nb_processes
                # if all simulations for this process were done, reassign to another
                # simulation
                if (current_sim_nb >= total_nb_simulations) & multi_core_opt:  # & (not new_simulation):
                    (current_sim_nb,
                     sim_id,
                     param_id,
                     core_id,
                     new_simulation) = _reassign_thread(thread_id,
                                                         total_nb_simulations,
                                                         thread_to_sim_id,
                                                         thread_masks,
                                                         timepoint_array,
                                                         min_time,
                                                         time_resolution,
                                                         times,
                                                         current_sim_nb,
                                                         sim_id, param_id,
                                                        core_id,
                                                         nb_parallel_cores,
                                                         nb_simulations,
                                                         start_nb_parallel_cores)
                if not new_simulation:
                    # print(111, thread_id, iteration_nb, sim_id,   core_id,
                    #       times[0, sim_id, param_id],
                    #       nb_parallel_cores[sim_id, param_id])
                    # if core_id == 0:
                    #     print(iteration_nb, sim_id)
                    break

            if success == 0:
                if core_id == 0:
                    print(969696969)
                break

            if success == 2:
                current_sim_nb += nb_processes
                # if core_id == 0:
                #     print(100000, iteration_nb)
                break

            # if (sim_id == 0) & (not re_assigned):
            #     if iteration_nb == 0:
            #         timepoint_array[0, sim_id, param_id] = math.floor(min_time / time_resolution[0])

            # if iteration_nb == 0:
            #     current_sim_nb += nb_processes
            #     if core_id == 0:
            #         print(3333, sim_id, times[0, sim_id, param_id])
            #     break

            iteration_nb += 1

        if success == 0:
            time_idx = 0
            while time_idx < (timepoint_array.shape[0] - 2):
                timepoint_array[time_idx+2, sim_id, param_id] = math.nan
                time_idx += 1
            break
        # current_sim_nb += nb_processes

    grid.sync()
    numba.cuda.syncthreads()

def _reassign_thread(thread_id,
                      total_nb_simulations, thread_to_sim_id,
                    thread_masks, timepoint_array, min_time,
                      time_resolution,
                      times, current_sim_nb, sim_id, param_id, core_id,
                      nb_parallel_cores,
                    nb_simulations, start_nb_parallel_cores):
    new_simulation = False
    # Reassign process to simulation of a process in the same warp

    warp_nb = thread_id // cuda.warpsize

    (new_sim_id,
     new_param_id,
     _,
     last_thread) = get_slowest_simulation_in_warp(thread_id, warp_nb,
                                                    total_nb_simulations,
                                                    thread_to_sim_id,
                                                    sim_id, param_id,
                                                    thread_masks,
                                                    timepoint_array,
                                                    min_time,
                                                    time_resolution, times)

    if new_sim_id != -1:
        # new_sim_id = slowest_sim_id
        # new_param_id = slowest_param_id
        # print(11111, thread_id)
        # print(thread_id, thread_to_sim_id[current_thread_nb, 0], sim_id,
        #       thread_to_sim_id[current_thread_nb, 1], param_id)
        # print(22222, thread_id)


        # cuda.atomic.add(thread_masks,
        #                 (1, new_sim_id, new_param_id),
        #                 thread_masks[0,sim_id,param_id])

        thread_masks[1, new_sim_id,
                     new_param_id] = (thread_masks[2,
                                                   sim_id,
                                                   param_id]
                                      +
                                      thread_masks[2,
                                                   new_sim_id,
                                                   new_param_id]
                                      )
        # print(888, thread_id, sim_id, param_id)

        # update sim id of current thread
        thread_to_sim_id[thread_id, 0] = new_sim_id
        thread_to_sim_id[thread_id, 1] = new_param_id

        cuda.syncwarp(thread_masks[0, sim_id, param_id])
        # update current sim nb based on new sim id
        current_sim_nb = (new_param_id * nb_simulations *
                          start_nb_parallel_cores +
                          new_sim_id * start_nb_parallel_cores)

        # no add one to the number of parallel cores of the
        # new sim id
        # This returns the number of cores before new ones were
        # added
        # How to get the new core_id, considering that
        # multiple new cores could be added
        # the number of added cores is the nb of parallel cores
        # of the old sim id
        # the mask of the old sim id defines which
        # warp thread idx were added
        # first sync newly added cores
        # cuda.syncwarp(thread_masks[0,sim_id, param_id])
        _get_core_ids = _get_incrementing_core_ids_for_reassigned_thread
        core_id = _get_core_ids(thread_to_sim_id,
                                 warp_nb,
                               thread_masks[0, sim_id, param_id],
                               nb_parallel_cores,
                               last_thread,
                               thread_id, new_sim_id, new_param_id)

        # cuda.syncwarp(thread_masks[0,sim_id, param_id])

        # print(thread_id, core_id)

        thread_to_sim_id[thread_id, 2] = core_id
        # create new mask by adding the mask of the old sim id
        # and the new sim id

        # if core_id >= nb_parallel_cores[new_sim_id, new_param_id]:
        #     print(1111, thread_id, core_id,
        #           nb_parallel_cores[sim_id, param_id])
        # if core_id == old_nb_cores:
        #     thread_masks[1,
        #                  new_sim_id,
        #                  new_param_id] += thread_masks[0,
        #                                              new_sim_id,
        #                                              new_param_id]
        sim_id = new_sim_id
        param_id = new_param_id
        new_simulation = True
    return current_sim_nb, sim_id, param_id, core_id, new_simulation

def get_slowest_simulation_in_warp(thread_id, warp_nb, total_nb_simulations,
                                   thread_to_sim_id, sim_id, param_id,
                                   thread_masks, timepoint_array,
                                   min_time, time_resolution, times):

    # warp_thread_idx = thread_id - warp_nb * cuda.warpsize
    current_thread_nb = warp_nb * cuda.warpsize
    last_thread = (warp_nb + 1) * cuda.warpsize
    last_thread = min(last_thread, total_nb_simulations)
    # check which simulation has progressed the least so far
    # (has the lowest elapsed time)
    lowest_sim_time = math.nan
    slowest_sim_id = -1
    slowest_param_id = -1
    use_first_sim_for_reassigning = False
    while current_thread_nb < last_thread:
        # don't consider switching to own thread_id
        if current_thread_nb != thread_id:
            current_sim_id = int(thread_to_sim_id[current_thread_nb, 0])
            current_param_id = int(thread_to_sim_id[current_thread_nb, 1])
            reassign = _check_whether_reassignment_is_allowed(current_sim_id,
                                                              current_param_id,
                                                               sim_id, param_id,
                                                               thread_masks,
                                                               timepoint_array,
                                                              min_time,
                                                               time_resolution)
            if (reassign) & ((current_sim_id != slowest_sim_id) |
                             (current_param_id != slowest_param_id)):
                # print(1111, sim_id, param_id, current_sim_id, current_param_id,
                #       times[0, current_sim_id, current_param_id],
                #       times[0, sim_id, param_id])
                sim_time = times[0, current_sim_id, current_param_id]
                if math.isnan(lowest_sim_time):
                    lowest_sim_time = sim_time
                    slowest_sim_id = current_sim_id
                    slowest_param_id = current_param_id
                elif(sim_time < lowest_sim_time):
                    lowest_sim_time = sim_time
                    slowest_sim_id = current_sim_id
                    slowest_param_id = current_param_id
                if use_first_sim_for_reassigning:
                    break
        current_thread_nb += 1

    return slowest_sim_id, slowest_param_id, lowest_sim_time, last_thread

def _check_whether_reassignment_is_allowed(current_sim_id, current_param_id,
                                           sim_id, param_id,
                                           thread_masks,
                                           timepoint_array, min_time,
                                           time_resolution):
    simulation_done = False
    if (timepoint_array[0, current_sim_id, current_param_id] >=
         math.floor(min_time/time_resolution[0])):
        simulation_done = True

    # if not simulation_done:
    #     print(1111, thread_id,
    #           timepoint_array[0, current_sim_id, current_param_id],
    #           math.floor(min_time/time_resolution[0]))
    # else:
    #     print(2222, thread_id,
    #           timepoint_array[0, current_sim_id, current_param_id],
    #           math.floor(min_time/time_resolution[0]))

    # make sure that process is added to a different simulation
    different_sim = False
    if ((current_sim_id != sim_id) |
            (current_param_id != param_id)):
        different_sim = True

    # if different_sim:
    #     diff_sim = 1
    # else:
    #     diff_sim = 0
    #
    # if simulation_done:
    #     sim_done = 1
    # else:
    #     sim_done = 0

    # print(333, current_thread_nb, thread_id, diff_sim, sim_done, sim_id, param_id,
    #       timepoint_array[0, current_sim_id, current_param_id])#
    # print(444,
    #       thread_masks[0, sim_id, param_id],
    #       thread_masks[1, sim_id, param_id],
    #       thread_masks[2, sim_id, param_id])

    if different_sim & (not simulation_done):
        # processes_currently_added_to_sim = False

        cuda.syncwarp(thread_masks[0, sim_id, param_id])

        # make sure that no other process is currently added
        # to this simulation
        # track this through thread_masks at 1 being non zero.
        # Once new processes are fully added to a simulation,
        # thread masks are set to 0 again.
        if (thread_masks[1, current_sim_id, current_param_id] ==
                thread_masks[2, current_sim_id, current_param_id]):
            # processes_currently_added_to_sim = True
            return True

    return False

def _get_incrementing_core_ids_for_reassigned_thread(thread_to_sim_id,
                                                         warp_nb,
                                                         thread_mask,
                                                         nb_parallel_cores,
                                                         last_thread,
                                                         thread_id, new_sim_id,
                                                         new_param_id):
    thread_to_sim_id[thread_id, 2] = math.nan
    cuda.syncwarp(thread_mask)

    # then go through all threads in the warp and
    # the first nan core ids will get the smaller new
    # core_ids
    new_core_id = (nb_parallel_cores[new_sim_id,
                                     new_param_id] - 1)
    new_core_id_thread_nb = warp_nb * cuda.warpsize
    while new_core_id_thread_nb < last_thread:
        # for each not assigned core id in the warp
        # increase the core id by one
        # check whether they work on the same simulation
        if ((thread_to_sim_id[thread_id, 0] == new_sim_id) &
                (thread_to_sim_id[thread_id, 1] == new_param_id)):
            if math.isnan(thread_to_sim_id[new_core_id_thread_nb,
                                           2]):
                new_core_id += 1
            # print(222, thread_to_sim_id[new_core_id_thread_nb,
            #                                2], new_core_id)
            # if the new core id thread number in the loop
            # is the thread number of the active process
            # then assign the core id
            if new_core_id_thread_nb == thread_id:
                break
        new_core_id_thread_nb += 1
    # print(333, core_id)
    cuda.syncwarp(thread_mask)
    return int(new_core_id)

def _execute_simulation_cpu(object_states, properties_array, times,
                        nb_simulations, nb_parameter_combinations,
                            parameter_value_array,
                            params_prop_dependence,
                                         position_dependence,
                            object_dependent_rates,
                            transition_parameters, all_transition_states,
                            action_parameters, action_state_array,
                        all_action_properties,action_operation_array,
                        current_transition_rates,
                        property_start_values,
                        property_extreme_values,
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
                            properties_tmax,
                            current_sum_tmax,

                            property_changes_tminmax_array,
                            property_changes_per_state,
                            nucleation_changes_per_state,
                            total_property_changes,
                            density_threshold_boundaries,

                            tau_square_eq_terms,

                            first_last_idx_with_object,
                                     local_object_lifetime_array,
                            local_lifetime_resolution,

                        timepoint_array,
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

    # HARD CODED TEST!
    some_creation_on_objects = True

    assertion_checks = True
    if thread_id < (nb_simulations * nb_parameter_combinations):
        last_transition = 0
        while True:
            success = _run_iteration(object_states,
                                     properties_array,
                                       times, parameter_value_array,
                            params_prop_dependence,
                                         position_dependence,
                            object_dependent_rates,
                                       transition_parameters,
                                     all_transition_states,
                                       action_parameters, action_state_array,
                                       all_action_properties,
                                     action_operation_array,
                                       current_transition_rates,
                                       property_start_values,
                                       property_extreme_values,
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
                                     properties_tmax,
                                     current_sum_tmax,

                                     property_changes_tminmax_array,
                                     property_changes_per_state,
                                    nucleation_changes_per_state,
                                     total_property_changes,
                                    density_threshold_boundaries,

                                     tau_square_eq_terms,

                                     first_last_idx_with_object,
                                     local_object_lifetime_array,
                                                 local_lifetime_resolution,

                                       timepoint_array, min_time,
                                       time_resolution, save_initial_state,

                                     nb_parallel_cores, thread_masks,

                                       local_density, local_resolution,
                                     total_density,
                                       rng_states, core_id, sim_id, param_id
                           )

            if (timepoint_array[0, sim_id, param_id] >=
                    math.floor(min_time/time_resolution[0])):
                break

            if success == 0:
                print("\n no success!")
                break

            if assertion_checks:
                # print("\n")
                # print(nb_objects_all_states[:,0,0])
                for state in range(1,np.max(object_states[0])+1):
                    state_mask = object_states[0] == state
                    # if state == 1:
                    #     plus_pos = (properties_array[0, 0][state_mask]
                    #                 + properties_array[0, 1][state_mask])
                    #     # print(current_transitions)
                    #     print(properties_array[0, 0][state_mask][plus_pos == 20])

                    plus_pos = (properties_array[0, 0][state_mask]
                                + properties_array[0, 1][state_mask]
                                + properties_array[0, 2][state_mask])
                    # print(current_transitions)
                    # print("\n", state)
                    # print(properties_array[0, 2][state_mask][plus_pos == 20])
                    # check that no property value of an object is nan
                    for property_nb in range(properties_array .shape[0]):
                        property_vals = properties_array[0, property_nb]
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
                    nb_objects = len(object_states[0][object_states[0] == state])
                    saved_nb_objects = nb_objects_all_states[0, state,
                                                             sim_id, param_id]
                    assert nb_objects == saved_nb_objects
                    # check that the maximum transition position is within
                    # the size of the array
                    max_position = all_transition_positions.max()
                    assert max_position < object_states[0].shape[0]
            last_transition = int(current_transitions[sim_id, param_id])

            iteration_nb += 1

        if success == 0:
            time_idx = 0
            while time_idx < (timepoint_array.shape[0]-2):
                timepoint_array[time_idx+2, sim_id, param_id] = math.nan
                time_idx += 1


def _run_iteration(object_states, properties_array , times,
                   parameter_value_array,
                            params_prop_dependence,
                                         position_dependence,
                            object_dependent_rates,
                   transition_parameters, all_transition_states,
                   action_parameters, action_state_array,
                   all_action_properties, action_operation_array,
                   current_transition_rates,
                   property_start_values,
                   property_extreme_values,
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
                   properties_tmax,
                   current_sum_tmax,
                    property_changes_tminmax_array,
                   property_changes_per_state, nucleation_changes_per_state,
                   total_property_changes,
                    density_threshold_boundaries,

                   tau_square_eq_terms,

                   first_last_idx_with_object,
                   local_object_lifetime_array,
                                                 local_lifetime_resolution,

                   timepoint_array, min_time, start_save_time,
                   time_resolution, save_initial_state,

                   nb_parallel_cores, thread_masks,

                   local_density, local_resolution, total_density,

                   rng_states, core_id, sim_id, param_id,
                   simulation_factor=None, parameter_factor=None
                   ):

    # Implement

    # if (core_id == 0) & (sim_id == 0):
    #     print(111, object_states[0,1,sim_id, param_id])
    #     # print(properties_array.shape[0],
    #     #       properties_array.shape[1],
    #     #       properties_array.shape[2],
    #     #       properties_array.shape[3],
    #     #       properties_array.shape[4])
    #     print(0, properties_array[0,0,1,sim_id, param_id])
    #     print(1, properties_array[0,1,1,sim_id, param_id])
    #     print(2, properties_array[0,2,1,sim_id, param_id])

    # create tensor for x (position in neurite), l (length of microtubule)
    # and time
    # start = time.time()
    # UPDATE nb_objects_all_states in each iteration at each iteration
    # thereby there is no need to go through all object states

    if some_creation_on_objects:

        _reset_local_density(local_density, nb_parallel_cores, core_id,
                             sim_id, param_id)
        if nb_parallel_cores[sim_id, param_id] > 1:
            cuda.syncwarp(thread_masks[0,sim_id, param_id])

        _get_local_and_total_density(local_density,
                                       total_density,
                                       local_resolution,
                                       first_last_idx_with_object,
                                       properties_array,
                                     property_extreme_values,
                                     creation_on_objects,
                                       object_states,
                                       nb_parallel_cores,thread_masks,
                                       core_id, sim_id, param_id)

        if nb_parallel_cores[sim_id, param_id] > 1:
            cuda.syncwarp(thread_masks[0,sim_id, param_id])

    # increase performance through an array for each object state
    # of which transitions are influenced by it
    # then don't recalculate the transition rates each iteration
    # but just update the few rates affected by the changed object state
    _get_rates = _get_total_and_single_rates_for_state_transitions
    _get_rates(parameter_value_array, params_prop_dependence, position_dependence,
                            object_dependent_rates,
               transition_parameters,
               all_transition_states, current_transition_rates, total_rates,
               nb_objects_all_states,creation_on_objects,
               total_density, local_resolution, object_states, properties_array,
               timepoint_array,
               property_extreme_values, first_last_idx_with_object,
               nb_parallel_cores,  thread_masks, core_id, sim_id, param_id,
               times)


    if nb_parallel_cores[sim_id, param_id] > 1:
        cuda.syncwarp(thread_masks[0,sim_id, param_id])

    if total_rates[sim_id, param_id] == 0:
        return 2

    # print(4567, total_rates[sim_id, param_id])

    if some_creation_on_objects:
    # if False:
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

        _get_tmin_tmax_for_property_changes(
                                             property_changes_tminmax_array,
                                             property_changes_per_state,
                                             total_property_changes,
                                            parameter_value_array,
                                            timepoint_array,
                                            property_extreme_values,
                                               current_sum_tmax,

                                               properties_tmax,
                                               properties_tmax_array,
                                               end_position_tmax_array,

                                               first_last_idx_with_object,
                                               object_states, properties_array ,

                                             nb_parallel_cores,
                                               core_id, sim_id, param_id
                                               )

        if nb_parallel_cores[sim_id, param_id] > 1:
            cuda.syncwarp(thread_masks[0,sim_id, param_id])

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

        # # The variables for quadratic equation are:
        tau = _get_tau(total_rates, first_last_idx_with_object, object_states,
                        property_changes_per_state,nucleation_changes_per_state,
                        property_changes_tminmax_array,
                       tau_square_eq_terms,
                         rng_states, simulation_factor, parameter_factor,
                         nb_parallel_cores,  thread_masks,
                       core_id, sim_id, param_id)

        # if sim_id == 28:
        #     print(tau)

        reaction_times[sim_id, param_id] = tau

        if nb_parallel_cores[sim_id, param_id] > 1:
            cuda.syncwarp(thread_masks[0,sim_id, param_id])

    else:

        get_reaction_times = _get_times_of_next_transition
        get_reaction_times(total_rates, reaction_times, sim_id, param_id,
                           rng_states, simulation_factor, parameter_factor)

    # check if a new timepoint should be saved
    # before updating object states and executing actions
    next_timepoint = max(start_save_time,
                         (timepoint_array[0, sim_id, param_id] *
                          time_resolution[0] +
                          time_resolution[0]))

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
                                    properties_array ,
                                nb_objects_all_states,
                                    property_extreme_values,

                                    local_object_lifetime_array,
                                                 local_lifetime_resolution,
                                    local_resolution,

                                    all_action_properties,
                                    action_operation_array,
                                    object_states,
                                    reaction_time_tmp,
                                    timepoint_array,
                                    first_last_idx_with_object,
                                    nb_parallel_cores,  core_id,
                                    sim_id, param_id,
                                    times)

        if nb_parallel_cores[sim_id, param_id] > 1:
            cuda.syncwarp(thread_masks[0,sim_id, param_id])

        if core_id == 0:
            times[0, sim_id, param_id] = (times[0, sim_id, param_id] +
                                          reaction_time_tmp)

        if nb_parallel_cores[sim_id, param_id] > 1:
            cuda.syncwarp(thread_masks[0,sim_id, param_id])

        _save_values_with_temporal_resolution(timepoint_array,
                                              times,
                                              object_states, properties_array ,
                                              time_resolution, start_save_time,
                                              save_initial_state,
                                              first_last_idx_with_object,
                                              nb_parallel_cores,
                                              thread_masks, core_id,
                                              sim_id, param_id)

        if core_id == 0:
            reaction_times[sim_id, param_id] = (reaction_times[sim_id, param_id] -
                                                reaction_time_tmp)

        if nb_parallel_cores[sim_id, param_id] > 1:
            cuda.syncwarp(thread_masks[0,sim_id, param_id])

        if times[0, sim_id, param_id] >= min_time:
            return 1

    _execute_actions_on_objects(parameter_value_array, action_parameters,
                                action_state_array,
                                properties_array ,
                                nb_objects_all_states,
                                property_extreme_values,

                                local_object_lifetime_array,
                                                 local_lifetime_resolution,
                                local_resolution,

                                all_action_properties,
                                action_operation_array,
                                object_states,
                                reaction_times[sim_id, param_id],
                                timepoint_array,
                                first_last_idx_with_object,
                                nb_parallel_cores,  core_id,
                                sim_id, param_id,
                                times)

    if nb_parallel_cores[sim_id, param_id] > 1:
        cuda.syncwarp(thread_masks[0,sim_id, param_id])

    if some_creation_on_objects:

        _reset_local_density(local_density, nb_parallel_cores, core_id,
                             sim_id, param_id)
        if nb_parallel_cores[sim_id, param_id] > 1:
            cuda.syncwarp(thread_masks[0,sim_id, param_id])

        # Update local and total density based on deterined tau
        _get_local_and_total_density(local_density,
                                       total_density,
                                       local_resolution,
                                       first_last_idx_with_object,
                                       properties_array ,
                                     property_extreme_values,
                                     creation_on_objects,
                                       object_states,
                                       nb_parallel_cores, thread_masks,
                                       core_id,
                                       sim_id, param_id)

        if nb_parallel_cores[sim_id, param_id] > 1:
            cuda.syncwarp(thread_masks[0,sim_id, param_id])

    # then get updated rates to have the correct nucleation rate and
    # total rate
    _get_rates = _get_total_and_single_rates_for_state_transitions
    _get_rates(parameter_value_array, params_prop_dependence, position_dependence,
                        object_dependent_rates,
               transition_parameters,
               all_transition_states, current_transition_rates, total_rates,
               nb_objects_all_states,creation_on_objects,
               total_density, local_resolution, object_states, properties_array,
               timepoint_array,
               property_extreme_values, first_last_idx_with_object,
               nb_parallel_cores,  thread_masks, core_id, sim_id, param_id,
               times)

    if core_id == 0:
        _determine_next_transition(total_rates, current_transition_rates,
                                   current_transitions, sim_id, param_id,
                                   rng_states, simulation_factor, parameter_factor)

    if nb_parallel_cores[sim_id, param_id] > 1:
        cuda.syncwarp(thread_masks[0,sim_id, param_id])

    if core_id == 0:
        # speed up searching for xth object with correct state
        # by keeping track of the position of each object in each state
        _determine_positions_of_transitions(current_transitions,
                                            all_transition_states,
                                            nb_objects_all_states,
                                            all_transition_positions,
                                            params_prop_dependence,
                                            object_dependent_rates,
                                            parameter_value_array,
                                            transition_parameters,
                                            current_transition_rates,
                                            timepoint_array,
                                            object_states, core_id,
                                            sim_id, param_id, rng_states,
                                            first_last_idx_with_object,
                                            simulation_factor, parameter_factor)

    if nb_parallel_cores[sim_id, param_id] > 1:
        cuda.syncwarp(thread_masks[0,sim_id, param_id])

    if math.isnan(all_transition_positions[sim_id, param_id]):
        if core_id == 0:
            print(all_transition_positions[sim_id, param_id],
                  sim_id, param_id, core_id,
                  nb_parallel_cores[sim_id, param_id])
        return 0

    _update_object_states(current_transitions, all_transition_states,
                          all_transition_positions, object_states,
                          nb_objects_all_states,
                          properties_array ,
                          transition_parameters,
                          all_transition_tranferred_vals,
                          all_transition_set_to_zero_properties,
                          property_start_values,
                            changed_start_values_array,
                            creation_on_objects,
                          first_last_idx_with_object,

                          local_object_lifetime_array,
                          local_lifetime_resolution,

                          parameter_value_array, params_prop_dependence,
                          property_extreme_values, timepoint_array,
                          current_transition_rates,

                          local_density, total_density,
                          local_resolution,
                          density_threshold_boundaries,
                          nb_parallel_cores, thread_masks, core_id,
                          sim_id, param_id,
                          rng_states, simulation_factor, parameter_factor,

                          times)

    if nb_parallel_cores[sim_id, param_id] > 1:
       cuda.syncwarp(thread_masks[0,sim_id, param_id])

    _remove_objects(all_object_removal_properties, object_removal_operations,
                    nb_objects_all_states, transition_parameters,
                    object_states, properties_array,
                    first_last_idx_with_object,
                    local_object_lifetime_array,
                                                 local_lifetime_resolution, times,
                    local_resolution,
                    nb_parallel_cores,  core_id, sim_id, param_id)

    if core_id == 0:
        times[0, sim_id, param_id] = (times[0, sim_id, param_id] +
                                      reaction_times[sim_id, param_id])

    if nb_parallel_cores[sim_id, param_id] > 1:
       cuda.syncwarp(thread_masks[0,sim_id, param_id])

    if math.isnan(times[0, sim_id, param_id]):
        if core_id == 0:
            print(sim_id, nb_objects_all_states[0, 0, sim_id, param_id])
        return 0

    thread_masks[0, sim_id, param_id] = 0
    return 1

def _get_random_number():
    pass

def _reassign_random_number_func_cpu():
    global _get_random_number
    _get_random_number = _get_random_number_cpu


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


    global _cut_object
    if not isinstance(_cut_object,
                      numba.core.registry.CPUDispatcher):
        _cut_object = numba.njit(
            _cut_object)

    global _track_object_property_changes
    if not isinstance(_track_object_property_changes,
                      numba.core.registry.CPUDispatcher):
        _track_object_property_changes = numba.njit(
            _track_object_property_changes)

    global _get_rate_of_density_dependent_transition
    if not isinstance(_get_rate_of_density_dependent_transition,
                      numba.core.registry.CPUDispatcher):
        _get_rate_of_density_dependent_transition = numba.njit(
            _get_rate_of_density_dependent_transition)

    global _get_rate_of_prop_dependent_transition
    if not isinstance(_get_rate_of_prop_dependent_transition,
                      numba.core.registry.CPUDispatcher):
        _get_rate_of_prop_dependent_transition = numba.njit(
            _get_rate_of_prop_dependent_transition)


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



    global _get_total_pos_dependent_creation_rate
    if not isinstance(_get_total_pos_dependent_creation_rate,
                      numba.core.registry.CPUDispatcher):
        _get_total_pos_dependent_creation_rate = numba.njit(
            _get_total_pos_dependent_creation_rate)

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

    global _get_density_dependent_position
    if not isinstance(_get_density_dependent_position,
                      numba.core.registry.CPUDispatcher):
        _get_density_dependent_position = numba.njit(
            _get_density_dependent_position)

    global _save_values_with_temporal_resolution
    if not isinstance(_save_values_with_temporal_resolution,
                      numba.core.registry.CPUDispatcher):
        _save_values_with_temporal_resolution = numba.njit(
            _save_values_with_temporal_resolution)

    global _remove_objects
    if not isinstance(_remove_objects,
                      numba.core.registry.CPUDispatcher):
        _remove_objects = numba.njit(_remove_objects)


def _decorate_all_functions_for_gpu(simulation_object, debug=False):
    debug = False
    opt = debug != True
    fastmath = False
    lineinfo = False

    global _get_random_number
    if not isinstance(_get_random_number,
                      numba.cuda.dispatcher.CUDADispatcher):
        _get_random_number = numba.cuda.jit(_get_random_number_gpu, debug=debug,
                                            opt=opt, fastmath=fastmath,
                                            lineinfo=lineinfo,
                                            device=True)

    global _run_iteration
    if not isinstance(_run_iteration,
                      numba.cuda.dispatcher.CUDADispatcher):
        _run_iteration = numba.cuda.jit(_run_iteration, debug=debug, opt=opt,
                                        fastmath=fastmath, lineinfo=lineinfo,
                                            device=True)

    global _cut_object
    if not isinstance(_cut_object,
                      numba.cuda.dispatcher.CUDADispatcher):
        _cut_object = numba.cuda.jit(
            _cut_object, debug=debug, opt=opt,
            fastmath=fastmath, lineinfo=lineinfo, device=True)

    global _track_object_property_changes
    if not isinstance(_track_object_property_changes,
                      numba.cuda.dispatcher.CUDADispatcher):
        _track_object_property_changes = numba.cuda.jit(
            _track_object_property_changes, debug=debug, opt=opt,
            fastmath=fastmath, lineinfo=lineinfo, device=True)

    global _get_rate_of_density_dependent_transition
    if not isinstance(_get_rate_of_density_dependent_transition,
                      numba.cuda.dispatcher.CUDADispatcher):
        _get_rate_of_density_dependent_transition = numba.cuda.jit(
            _get_rate_of_density_dependent_transition, debug=debug, opt=opt,
            fastmath=fastmath, lineinfo=lineinfo,
                                            device=True)

    global _get_rate_of_prop_dependent_transition
    if not isinstance(_get_rate_of_prop_dependent_transition,
                      numba.cuda.dispatcher.CUDADispatcher):
        _get_rate_of_prop_dependent_transition = numba.cuda.jit(
            _get_rate_of_prop_dependent_transition, debug=debug, opt=opt,
            fastmath=fastmath, lineinfo=lineinfo,
                                            device=True)

    global _reassign_thread
    if not isinstance(_reassign_thread,
                      numba.cuda.dispatcher.CUDADispatcher):
        _reassign_thread = numba.cuda.jit(
            _reassign_thread, debug=debug, opt=opt, fastmath=fastmath,
            lineinfo=lineinfo,
                                            device=True)

    global get_slowest_simulation_in_warp
    if not isinstance(get_slowest_simulation_in_warp,
                      numba.cuda.dispatcher.CUDADispatcher):
        get_slowest_simulation_in_warp = numba.cuda.jit(
            get_slowest_simulation_in_warp, debug=debug, opt=opt,
            fastmath=fastmath, lineinfo=lineinfo,
                                            device=True)

    global _check_whether_reassignment_is_allowed
    if not isinstance(_check_whether_reassignment_is_allowed,
                      numba.cuda.dispatcher.CUDADispatcher):
        _check_whether_reassignment_is_allowed = numba.cuda.jit(
            _check_whether_reassignment_is_allowed, debug=debug, opt=opt,
            fastmath=fastmath, lineinfo=lineinfo,
                                            device=True)

    global _get_incrementing_core_ids_for_reassigned_thread
    if not isinstance(_get_incrementing_core_ids_for_reassigned_thread,
                      numba.cuda.dispatcher.CUDADispatcher):
        _get_incrementing_core_ids_for_reassigned_thread = numba.cuda.jit(
            _get_incrementing_core_ids_for_reassigned_thread, debug=debug, opt=opt,
            fastmath=fastmath, lineinfo=lineinfo,
                                            device=True)

    global _get_nucleation_on_objects_rate
    if not isinstance(_get_nucleation_on_objects_rate,
                      numba.cuda.dispatcher.CUDADispatcher):
        _get_nucleation_on_objects_rate = numba.cuda.jit(
            _get_nucleation_on_objects_rate, debug=debug, opt=opt,
            fastmath=fastmath, lineinfo=lineinfo,
                                            device=True)

    global _get_nucleation_changes_per_state
    if not isinstance(_get_nucleation_changes_per_state,
                      numba.cuda.dispatcher.CUDADispatcher):
        _get_nucleation_changes_per_state = numba.cuda.jit(
            _get_nucleation_changes_per_state, debug=debug, opt=opt,
            fastmath=fastmath, lineinfo=lineinfo,
                                            device=True)

    global _get_first_and_last_object_pos
    if not isinstance(_get_first_and_last_object_pos,
                      numba.cuda.dispatcher.CUDADispatcher):
        _get_first_and_last_object_pos = numba.cuda.jit(
            _get_first_and_last_object_pos, debug=debug, opt=opt,
            fastmath=fastmath, lineinfo=lineinfo,
                                            device=True)

    global _get_tau
    if not isinstance(_get_tau,
                      numba.cuda.dispatcher.CUDADispatcher):
        _get_tau = numba.cuda.jit(_get_tau, debug=debug, opt=opt,
                                  fastmath=fastmath, lineinfo=lineinfo,
                                            device=True)

    global _reset_local_density
    if not isinstance(_reset_local_density,
                      numba.cuda.dispatcher.CUDADispatcher):
        _reset_local_density = numba.cuda.jit(
        _reset_local_density, debug=debug, opt=opt, fastmath=fastmath,
            lineinfo=lineinfo,
                                            device=True)

    global _get_local_and_total_density
    if not isinstance(_get_local_and_total_density,
                      numba.cuda.dispatcher.CUDADispatcher):
        _get_local_and_total_density = numba.cuda.jit(
        _get_local_and_total_density, debug=debug, opt=opt, fastmath=fastmath,
            lineinfo=lineinfo,
                                            device=True)

    global _get_total_pos_dependent_creation_rate
    if not isinstance(_get_total_pos_dependent_creation_rate,
                      numba.cuda.dispatcher.CUDADispatcher):
        _get_total_pos_dependent_creation_rate = numba.cuda.jit(
            _get_total_pos_dependent_creation_rate, debug=debug,
            opt=opt,
            fastmath=fastmath, lineinfo=lineinfo,
            device=True)

    global _get_total_and_single_rates_for_state_transitions
    if not isinstance(_get_total_and_single_rates_for_state_transitions,
                      numba.cuda.dispatcher.CUDADispatcher):
        _get_total_and_single_rates_for_state_transitions = numba.cuda.jit(
        _get_total_and_single_rates_for_state_transitions, debug=debug, opt=opt,
            fastmath=fastmath, lineinfo=lineinfo,
                                            device=True)


    global _get_property_action_changes
    if not isinstance(_get_property_action_changes,
                      numba.cuda.dispatcher.CUDADispatcher):
        _get_property_action_changes = numba.cuda.jit(
        _get_property_action_changes, debug=debug, opt=opt, fastmath=fastmath,
            lineinfo=lineinfo,
                                            device=True)

    global _get_tmin_tmax_for_property_changes
    if not isinstance(_get_tmin_tmax_for_property_changes,
                      numba.cuda.dispatcher.CUDADispatcher):
        _get_tmin_tmax_for_property_changes = numba.cuda.jit(
            _get_tmin_tmax_for_property_changes, debug=debug,
            opt=opt, fastmath=fastmath, lineinfo=lineinfo,
                                            device=True)

    global _get_times_of_next_transition
    if not isinstance(_get_times_of_next_transition,
                      numba.cuda.dispatcher.CUDADispatcher):
        _get_times_of_next_transition = numba.cuda.jit(
            _get_times_of_next_transition, debug=debug, opt=opt,
            fastmath=fastmath, lineinfo=lineinfo,
                                            device=True)

    global _determine_next_transition
    if not isinstance(_determine_next_transition,
                      numba.cuda.dispatcher.CUDADispatcher):
        _determine_next_transition = numba.cuda.jit(_determine_next_transition,
                                                    debug=debug, opt=opt,
                                                    fastmath=fastmath,
                                                    lineinfo=lineinfo,
                                            device=True)

    global _determine_positions_of_transitions
    if not isinstance(_determine_positions_of_transitions,
                      numba.cuda.dispatcher.CUDADispatcher):
        _determine_positions_of_transitions = numba.cuda.jit(
            _determine_positions_of_transitions, debug=debug, opt=opt,
            fastmath=fastmath, lineinfo=lineinfo,
                                            device=True)

    global _execute_actions_on_objects
    if not isinstance(_execute_actions_on_objects,
                      numba.cuda.dispatcher.CUDADispatcher):
        _execute_actions_on_objects = numba.cuda.jit(
            _execute_actions_on_objects, debug=debug, opt=opt,
            fastmath=fastmath, lineinfo=lineinfo,
                                            device=True)

    global _increase_lowest_no_object_idx
    if not isinstance(_increase_lowest_no_object_idx,
                      numba.cuda.dispatcher.CUDADispatcher):
        _increase_lowest_no_object_idx = numba.cuda.jit(
            _increase_lowest_no_object_idx, debug=debug, opt=opt,
            fastmath=fastmath, lineinfo=lineinfo,
                                            device=True)

    global _reduce_highest_object_idx
    if not isinstance(_reduce_highest_object_idx,
                      numba.cuda.dispatcher.CUDADispatcher):
        _reduce_highest_object_idx = numba.cuda.jit(_reduce_highest_object_idx,
                                               debug=debug, opt=opt,
                                                    fastmath=fastmath,
                                                 lineinfo=lineinfo,
                                            device=True)

    global _get_density_dependent_position
    if not isinstance(_get_density_dependent_position,
                      numba.cuda.dispatcher.CUDADispatcher):
        _get_density_dependent_position = numba.cuda.jit(
            _get_density_dependent_position,
            debug=debug, opt=opt,
            fastmath=fastmath,
            lineinfo=lineinfo,
                                            device=True)

    global _update_object_states
    if not isinstance(_update_object_states,
                      numba.cuda.dispatcher.CUDADispatcher):
        _update_object_states = numba.cuda.jit(_update_object_states,
                                               debug=debug, opt=opt,
                                               fastmath=fastmath,
                                                 lineinfo=lineinfo,
                                            device=True)


    global _get_random_object_at_position
    if not isinstance(_get_random_object_at_position,
                      numba.cuda.dispatcher.CUDADispatcher):
        _get_random_object_at_position = numba.cuda.jit(
            _get_random_object_at_position, debug=debug, opt=opt,
            fastmath=fastmath, lineinfo=lineinfo, device=True)

    global _save_values_with_temporal_resolution
    if not isinstance(_save_values_with_temporal_resolution,
                      numba.cuda.dispatcher.CUDADispatcher):
        _save_values_with_temporal_resolution = numba.cuda.jit(
            _save_values_with_temporal_resolution, debug=debug, opt=opt,
            fastmath=fastmath, lineinfo=lineinfo,
                                            device=True)

    global _remove_objects
    if not isinstance(_remove_objects,
                      numba.cuda.dispatcher.CUDADispatcher):
        _remove_objects = numba.cuda.jit(_remove_objects, debug=debug, opt=opt,
                                         fastmath=fastmath, lineinfo=lineinfo,
                                            device=True)

    global _execute_simulation_gpu
    if not isinstance(_execute_simulation_gpu,
                      numba.cuda.dispatcher.CUDADispatcher):
        # get all argument names of function
        # args = _execute_simulation_gpu.__code__.co_varnames
        # args = inspect.signature(_execute_simulation_gpu).parameters
        # signature_list = []
        # for arg_nb, arg in enumerate(args):
        #     arg_type = numba.typeof(getattr(simulation_object, arg))
        #     # print(arg, arg_type)
        #     # print(dir(arg_type))
        #     # print(arg_type.name)
        #     if hasattr(arg_type, "dtype"):
        #         # print(arg_type.layout)
        #         dimensions = arg_type.key[1]
        #         arg_type_str = str(arg_type.dtype)+"["
        #         for dimension in range(dimensions):
        #             arg_type_str+= ":"
        #             if dimension < (dimensions - 1):
        #                 arg_type_str+=","
        #         arg_type_str += "]"
        #     else:
        #         arg_type_str = str(arg_type)
        #     if arg_type_str == "bool":
        #         arg_type_str = "boolean"
        #     if arg_type_str.find("Record") != -1:
        #         arg_type_str = "float32[:]"
        #     signature_list.append(arg_type_str)
        #
        # signature_string = "void("+",".join(signature_list)+")"
        # # print(numba.core.types.__dict__)
        # # print(numba.core.sigutils._parse_signature_string(signature_string))
        # # dasd
        # print(signature_string)
        # _execute_simulation_gpu_func = numba.cuda.jit(signature_string,
        #                                          debug=debug, opt=opt,
        #                                          fastmath=fastmath)
        # print("sig done")
        # _execute_simulation_gpu= _execute_simulation_gpu_func(_execute_simulation_gpu)
        # print("compiled!")
        _execute_simulation_gpu = numba.cuda.jit(_execute_simulation_gpu,
                                                 debug=debug, opt=opt,
                                                 fastmath=fastmath,
                                                 lineinfo=lineinfo)

def _get_nucleation_on_objects_rate(creation_on_objects,
                                    transition_parameters,
                                    parameter_value_array,
                                    core_id, sim_id, param_id):
    # get nucleation on objects rate
    transition_nb = 0
    transition_nb_creation_on_objects = math.nan
    while transition_nb < creation_on_objects.shape[0]:
        if (not math.isnan(creation_on_objects[transition_nb, 0, 0])):
            transition_nb_creation_on_objects = transition_nb
            break
        transition_nb += 1

    if math.isnan(transition_nb_creation_on_objects):
        return math.nan

    nucleation_on_objects_rate = parameter_value_array[
        int(transition_parameters[int(transition_nb_creation_on_objects),0]),
        0, param_id]

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
        action_states = action_state_array[action_nb]
        action_property_nb = int(all_action_properties[action_nb, 0])
        action_val_sign = (action_operation_array[action_nb] *
                           parameter_value_array[int(action_parameters[action_nb]),
                                                 0, param_id])
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
            while action_state_nb < action_states.shape[0]:
                if math.isnan(action_states[action_state_nb]):
                    break
                property_changes_per_state[int(action_states[action_state_nb])-1,
                                           action_property_nb,
                                           sim_id, param_id] += action_val_sign
                total_property_changes[int(action_states[action_state_nb])-1,
                                       sim_id, param_id] += action_val_sign
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
            nucleation_changes_per_state[state_nb,
                                        property_nb,
                                         sim_id,
                                         param_id] = (property_changes_per_state[state_nb,
                                                                                 property_nb,
                                                                                 sim_id,
                                                                                 param_id] *
                                                      nucleation_on_objects_rate)
            property_nb += 1
        state_nb += 1

def _get_first_and_last_object_pos(nb_objects, nb_parallel_cores, core_id):
    objects_per_core = math.ceil(nb_objects / nb_parallel_cores)
    if core_id < 0:
        core_id_use = int(- (core_id + 1))
    else:
        core_id_use = int(core_id)

    object_pos = objects_per_core * core_id_use
    last_object_pos = objects_per_core * (core_id_use + 1)
    last_object_pos = min(last_object_pos, nb_objects)
    # if core_id is negative, then it should be started from the back
    # (negative object positions)
    if core_id < 0:
        last_object_pos = int(- last_object_pos)
        if object_pos == 0:
            first_object_pos = -2
        else:
            first_object_pos = - object_pos
    else:
        first_object_pos = object_pos
    return int(first_object_pos), int(last_object_pos)
#
# def _get_first_and_last_object_pos(nb_objects, nb_parallel_cores, core_id):
#     objects_per_core = math.ceil(nb_objects / nb_parallel_cores)
#     if core_id < 0:
#         core_id_use = int(- (core_id + 1))
#     else:
#         core_id_use = int(core_id)
#
#     object_pos = math.floor(objects_per_core * core_id_use)
#     last_object_pos = math.floor(objects_per_core * (core_id_use + 1))
#     last_object_pos = min(last_object_pos, nb_objects)
#     # if core_id is negative, then it should be started from the back
#     # (negative object positions)
#     if core_id < 0:
#         last_object_pos = int(- last_object_pos)
#         if object_pos == 0:
#             first_object_pos = -2
#         else:
#             first_object_pos = - object_pos
#     else:
#         first_object_pos = object_pos
#     return int(first_object_pos), int(last_object_pos)

def _reset_local_density(local_density, nb_parallel_cores, core_id,
                         sim_id, param_id):
    nb_x_pos = local_density.shape[1]
    (x_pos,
     last_x_pos) = _get_first_and_last_object_pos(nb_x_pos,
                                                  nb_parallel_cores[sim_id,
                                                                    param_id],
                                                  core_id)
    while x_pos < last_x_pos:
        property_nb = 0
        while property_nb < local_density.shape[0]:
            local_density[property_nb, x_pos, sim_id, param_id] = 0
            property_nb += 1
        x_pos += 1


# def _get_local_and_total_density(local_density, total_density, local_resolution,
#                                  first_last_idx_with_object, properties_array ,
#                                  object_states, nb_parallel_cores,
#                                  thread_masks,
#                                  core_id, sim_id, param_id):
#     # get density of MTs at timepoint
#     if core_id == 0:
#         total_density[sim_id, param_id] = 0
#
#     if nb_parallel_cores[sim_id, param_id] > 1:
#         cuda.syncwarp(int(thread_masks[0, sim_id, param_id]))
#
#     nb_objects = first_last_idx_with_object[1,sim_id, param_id] + 1
#     (object_pos,
#      last_object_pos) = _get_first_and_last_object_pos(nb_objects,
#                                                        nb_parallel_cores[
#                                                            sim_id, param_id],
#                                                        core_id)
#
#     while object_pos < last_object_pos:
#         if object_states[object_pos, sim_id, param_id] > 0:
#
#             start = properties_array[0, 0, object_pos, sim_id, param_id]
#             end = start + properties_array[0, 1, object_pos, sim_id, param_id]
#             end += properties_array[0, 2, object_pos, sim_id, param_id]
#             if start != end:
#                 x_start = int(math.floor(start / local_resolution))
#                 x_end = int(math.ceil(end / local_resolution))
#                 position_at_end = False
#                 if x_end > local_density.shape[1]:
#                     x_end = local_density.shape[1]
#                     position_at_end = True
#                 x_pos = max(x_start, 0)
#                 while x_pos < x_end:
#                     if (x_pos == (x_end - 1)) & (not position_at_end):
#                         # for the x bin in which the MT ended, don't add a full MT
#                         # but just the relative amount of the bin crossed by the MT
#                         x_um = x_pos * local_resolution
#                         diff = (end - x_um) / local_resolution
#                         cuda.atomic.add(local_density,
#                                         (0, x_pos, sim_id, param_id), diff)
#                         cuda.atomic.add(total_density,
#                                         (sim_id, param_id), diff)
#                     else:
#                         cuda.atomic.add(local_density,
#                                         (0, x_pos, sim_id, param_id), 1)
#                         cuda.atomic.add(total_density,
#                                         (sim_id, param_id), 1)
#                     x_pos += 1
#         object_pos += 1

def _get_local_and_total_density(local_density, total_density, local_resolution,
                                 first_last_idx_with_object, properties_array ,
                                 property_extreme_values, creation_on_objects,
                                 object_states, nb_parallel_cores,
                                 thread_masks,
                                 core_id, sim_id, param_id):
    # get density of MTs at timepoint
    if core_id == 0:
        property_nb = 0
        while property_nb <= total_density.shape[0]:
            total_density[property_nb, sim_id, param_id] = 0
            property_nb += 1

    if nb_parallel_cores[sim_id, param_id] > 1:
        cuda.syncwarp(thread_masks[0,sim_id, param_id])

    nb_objects = first_last_idx_with_object[1,sim_id, param_id] + 1
    (object_pos,
     last_object_pos) = _get_first_and_last_object_pos(
        nb_objects, nb_parallel_cores[sim_id, param_id], core_id)
    # if sim_id == 356:
    #     print(1111, core_id, object_pos, last_object_pos)
    #
    # cuda.syncwarp(thread_masks[0,sim_id, param_id])
    # if sim_id == 356:
    #     print(2222, core_id, object_pos, last_object_pos)

    while object_pos < last_object_pos:
        if object_states[0, object_pos, sim_id, param_id] > 0:

            creation_property = 0
            end = 0
            while creation_property < creation_on_objects.shape[2]:
                # stop calculation of densities of individual properties
                # once there is a nan in creation on objects
                # in the case of only nans, only the total density is needed
                if ((creation_property > 0) &
                        (math.isnan(creation_on_objects[
                                        -1, 0, creation_property]))):
                    break

                # don't perform calculations if object has zero value of
                # needed property
                # but only for the last property of the array since
                # calculations depend on calculations of previous properties
                # since calculations are started from end point of last
                # calculation
                if creation_property == (creation_on_objects.shape[0] -1):
                    last_idx = 1
                else:
                    last_idx = int(math.isnan(creation_on_objects[
                                                    -1, 0,
                                                    creation_property + 1]))

                # last_idx = cuda.selp(creation_property ==
                #                      (creation_on_objects.shape[0] -1),
                #
                #                      1,
                #                      int(math.isnan(creation_on_objects[
                #                                     -1, 0,
                #                                     creation_property + 1]))
                #                      )
                # print(222)
                if (((math.isnan(creation_on_objects[-1, 0, 0])) |
                        (properties_array[0, int(creation_on_objects[
                                                     -1, 0,
                                                     creation_property]),
                                        object_pos, sim_id, param_id] > 0))
                    | (last_idx == 0)):
                    # creation_on_objects defines which property is actually
                    # used while the start property defines the end of which
                    # property defines the start position

                    # if only the total local density is needed,
                    # the start property is 0

                    # Due to property numbers in creation_on_objects[-1
                    # being sorted ascendingly, the last end can be used as
                    # new start. Then all properties from the
                    # last end property + 1 to the new start property
                    # have to still be added.

                    if math.isnan(creation_on_objects[-1, 0, 0]):
                        target_property = 0
                    else:
                        target_property = int(creation_on_objects[-1, 0, creation_property])

                    # target_property = cuda.selp(
                    #     math.isnan(creation_on_objects[-1, 0, 0]),
                    #
                    #     0,
                    #     int(creation_on_objects[-1, 0, creation_property]))

                    # start either at property_nb of 0
                    # or start at one after the last quantified
                    # property_nb + 1
                    if creation_property == 0:
                        property_nb = 0
                    else:
                        property_nb = int(creation_on_objects[
                                                    -1, 0, creation_property
                                                    - 1] + 1)
                    # property_nb = cuda.selp(creation_property == 0,
                    #
                    #                         0,
                    #                         int(creation_on_objects[
                    #                                 -1, 0, creation_property
                    #                                 - 1] + 1)
                    #                         )
                    # use last end as new start
                    start = end
                    while property_nb <= max(0, target_property-1):
                        start += properties_array[0, property_nb,
                                                  object_pos,
                                                  sim_id, param_id]
                        property_nb += 1

                    # print(1, property_nb, target_property, start, end)
                    # to get the end value of the property add all
                    # properties until the defined property value that should
                    # be quantified - this is always just one property
                    # further, except for calculating the combined density
                    # of all properties only. In that case, it is all the
                    # properties
                    end = start
                    # the end property number is the last property number
                    # if the combined local density should be calculated
                    # and otherwise the  property number defined in
                    # creation_on_objects
                    if math.isnan(creation_on_objects[-1, 0,
                                                      creation_property]):
                        max_property = properties_array.shape[1] - 1
                    else:
                        max_property = int(creation_on_objects[-1, 0,
                                                    creation_property])
                    # max_property = cuda.selp(math.isnan(
                    #         creation_on_objects[-1, 0, creation_property]),
                    #
                    #         properties_array.shape[1] - 1,
                    #
                    #         int(creation_on_objects[-1, 0,
                    #                                 creation_property]))

                    while property_nb <= max_property:
                        end += properties_array[0, property_nb,
                                                object_pos,
                                                sim_id, param_id]
                        property_nb += 1

                    # print(2, max_property, end)
                    # minimum allowed start position is negative local_resolution
                    # since position 0 is the amount of MTs from
                    # - local_resolution to 0
                    # while the last position, is the amount of MTs from
                    # x_max to local_resolution to x_max
                    # start += properties_array[0, 1,object_pos,sim_id, param_id]
                    # end = start + properties_array[0, 1,object_pos,sim_id, param_id]
                    # end += properties_array[0, 2,object_pos,sim_id, param_id]
                    # if not math.isnan(property_extreme_values[1, 0, 0]):
                    #     end = min(property_extreme_values[1, 0, 0], end)
                    # start = max(start, -local_resolution)
                    start = max(start, 0)
                    # x_start = int(math.ceil(start / local_resolution))
                    x_start = int(math.floor(start / local_resolution))
                    if ((start != end) & (end > 0) &
                            (x_start <= local_density.shape[1] - 1)):
                        # x_end = int(math.ceil(end / local_resolution))
                        x_end = int(math.floor(end / local_resolution))
                        # for MTs going beyond the end (in the case of an open end)
                        # take the last position possible as the end position
                        x_end = min(x_end, local_density.shape[1] - 1)
                        x_pos = max(x_start, 0)
                        first_x = True
                        # track the cumulative density of this object
                        # to get the cumulative local density
                        # density_from_object = 0
                        while x_pos <= x_end:
                            density = 1

                            # density -= cuda.selp(first_x,
                            #
                            #                      (1 -
                            #                       (x_pos * local_resolution - start)
                            #                       / local_resolution),
                            #
                            #                      0.0
                            #                      )
                            #
                            # density -= cuda.selp((x_pos == x_end),
                            #
                            #                      ((x_pos * local_resolution - end) / local_resolution),
                            #
                            #                      0.0
                            #                      )
                            if first_x:
                                # the start point is not actually at the very
                                # beginning of the resolution but after that
                                # therefore the added density is below 1 for the
                                # first position. Subtract the relative amount of
                                # the local resolution that the object starts
                                # after the x_pos
                                first_x = False
                                # x_um = x_pos * local_resolution
                                density -= ((start -
                                            x_pos * local_resolution) /
                                            local_resolution)
                                # density -= ( 1 -
                                #              (x_pos * local_resolution - start)
                                #              / local_resolution)
                            if (x_pos == x_end):
                                # for the x bin in which the MT ended, don't add a
                                # full MT but just the relative amount of the bin
                                # crossed by the MT
                                # x_um = x_pos * local_resolution
                                # density -= ((x_pos * local_resolution - end) / local_resolution)
                                density -= (1 - ((end - x_pos *
                                                  local_resolution) /
                                                 local_resolution))
                            # multiply by local_resolution to get the actual
                            # density in objects/um
                            density *= local_resolution
                            # first_x = False
                            # if density > 1:
                            #     print(1111111, x_pos, density)

                            if x_pos >= local_density.shape[1]:
                                print(77770)

                            # if target_property >= local_density.shape[0]:
                            #     print(77771)

                            cuda.atomic.add(local_density,
                                            (target_property, x_pos,
                                             sim_id, param_id), density)
                            # check whether x pos is not higher than
                            # local density size (still within neurite)
                            # if x_pos > local_density.shape[1]:
                            #     print(1111111, x_pos, local_density.shape[1],
                            #           properties_array[0, 0,
                            #                            object_pos,
                            #                            sim_id, param_id],
                            #           properties_array[0, 1,
                            #                            object_pos,
                            #                            sim_id, param_id],
                            #           properties_array[0, 2,
                            #                            object_pos,
                            #                            sim_id, param_id]
                            #           )
                            #
                            # # check whether density is always above 0
                            # if (density < 0):
                            #     print(7777777, density, object_pos,
                            #           x_um, start, end
                            #           )
                            # # check whether end is never lower than start
                            # # (which would indicate negative properties)
                            # if end < start:
                            #     print(666666, start, end, density)

                            # density_from_object += density
                            x_pos += 1

                        # cuda.atomic.add(total_density,
                        #                 (target_property, sim_id, param_id),
                        #                 density_from_object)
                creation_property += 1
        object_pos += 1

    if nb_parallel_cores[sim_id, param_id] > 1:
        cuda.syncwarp(thread_masks[0,sim_id, param_id])

    # summing local density up afterwards for total_density
    # is faster then adding to total_density
    # while filling the local density above
    nb_pos = local_density.shape[1]
    (x_pos, last_x_pos) = _get_first_and_last_object_pos(
        nb_pos, nb_parallel_cores[sim_id, param_id], core_id)

    while x_pos < last_x_pos:
        property_nb = 0
        while property_nb < local_density.shape[0]:
            cuda.atomic.add(total_density,
                            (property_nb, sim_id, param_id),
                            local_density[property_nb, x_pos,
                                          sim_id, param_id])
            property_nb += 1
        x_pos += 1

    # # Test whether all values in local density together sum up to total
    # # density
    # if nb_parallel_cores[sim_id, param_id] > 1:
    #     cuda.syncwarp(thread_masks[0,sim_id, param_id])
    #
    # if core_id == 0:
    #     property_nb = 0
    #     while property_nb < total_density.shape[0]:
    #         density_sum = 0
    #         x_pos = 0
    #         while x_pos < local_density.shape[1]:
    #             density_sum += local_density[property_nb, x_pos,
    #                                          sim_id, param_id]
    #             x_pos += 1
    #
    #         if round(density_sum, 4) < round(total_density[property_nb,
    #                                                        sim_id,
    #                                                        param_id],4):
    #             print(9999999, density_sum, total_density[property_nb,
    #                                                       sim_id, param_id],
    #                   sim_id, property_nb, target_property)
    #
    #         if round(density_sum, 4) > round(total_density[property_nb,
    #                                                        sim_id,
    #                                                        param_id],
    #                                          4):
    #             print(8888888, density_sum, total_density[property_nb,
    #                                                       sim_id, param_id],
    #                   sim_id, property_nb, target_property)
    #         property_nb += 1

def _get_total_and_single_rates_for_state_transitions(parameter_value_array,
                                                      params_prop_dependence,
                                                      position_dependence,
                                                      object_dependent_rates,
                                                      transition_parameters,
                                                      all_transition_states,
                                                      current_transition_rates,
                                                      total_rates,
                                                      nb_objects_all_states,
                                                      creation_on_objects,
                                                      total_density,
                                                      local_resolution,
                                                      object_states,
                                                      properties_array,
                                                      timepoint_array,
                                                      property_extreme_values,
                                                      first_last_idx_with_object,
                                                      nb_parallel_cores,
                                                      thread_masks,
                                                      core_id, sim_id, param_id,
                                                      times):

    if core_id == 0:
        total_rates[sim_id, param_id] = 0

    if nb_parallel_cores[sim_id, param_id] > 1:
        cuda.syncwarp(thread_masks[0,sim_id, param_id])

    nb_transitions = transition_parameters.shape[0]
    nb_cores_sim = nb_parallel_cores[sim_id, param_id]
    # get rates for all state transitions, depending on number of objects
    # in corresponding start state of transition
    (transition_nb,
     last_transition_nb) = _get_first_and_last_object_pos(nb_transitions,
                                                          nb_cores_sim,
                                                          core_id)

    while transition_nb < last_transition_nb:
        transition_states = all_transition_states[transition_nb]
        start_state = transition_states[0]
        transition_rate = parameter_value_array[int(transition_parameters[
                                                        transition_nb, 0]),
                                                int(timepoint_array[
                                                        0, sim_id, param_id]),
                                                param_id]
        if start_state == 0:
            # for state 0, the number of objects in state 0 does not matter
            if not math.isnan(creation_on_objects[transition_nb, 0, 0]):
                density_sum = 0
                # if only one density was measured (of all combined properties)
                # then only take the first index in total density
                if math.isnan(creation_on_objects[-1, 0, 0]):
                    density_sum += total_density[0, sim_id, param_id]
                else:
                    # Otherwise sum all densities relevant to the current
                    # transition
                    property_nb = 0
                    while property_nb < creation_on_objects.shape[2]:
                        if math.isnan(creation_on_objects[transition_nb, 0,
                                                          property_nb]):
                            break
                        density_sum += total_density[int(creation_on_objects[
                            transition_nb, 0, property_nb]), sim_id, param_id]
                        property_nb += 1

                current_transition_rates[transition_nb,
                                         sim_id, param_id] = (transition_rate *
                                                              density_sum)

            elif not math.isnan(params_prop_dependence[
                                    int(transition_parameters[transition_nb, 0])
                                    , 2]):
                # POSITION DEPENDENT NUCLEATION IS ONLY IMPLEMENTED
                # FOR LINEAR POSITION DEPENDENCE SO FAR!!!

                # calculate the total rate from inside the region
                # where the rate depends on the position
                (total_dependence_rate, max_property_change,
                 max_position,_,_,_) = _get_total_pos_dependent_creation_rate(
                    transition_nb,
                    params_prop_dependence[
                        int(transition_parameters[transition_nb, 0])],
                    parameter_value_array,
                    property_extreme_values,
                    transition_parameters,
                    timepoint_array, sim_id, param_id,
                )

                # then add to this total_dependence_rate the rate from outside
                # the dependence by multiplying the remaining region with
                # the baseline rate
                total_dependence_rate += ((max_position - max_property_change) *
                                          (transition_rate/max_position))
                current_transition_rates[transition_nb,
                                         sim_id,
                                         param_id] = total_dependence_rate
            else:
                current_transition_rates[transition_nb,
                                         sim_id, param_id] = transition_rate

            # for nucleation multiply the nucleation rate by the fraction of
            # free nucleation sites if limited resources are defined.
            # Resources are implemented as parameter, which is why the value of
            # the limit has to be obtained from the respective parameter
            # Thereby implement a resource limitation for nucleating new objects
            if not math.isnan(transition_parameters[transition_nb, 1]):
                # multiply by the percentage of unused resources.
                # If all resources are used, the transition rate is 0

                current_transition_rates[transition_nb,
                                         sim_id,
                                         param_id] *= (
                        (parameter_value_array[int(transition_parameters[
                                                       transition_nb, 1]),
                                               int(timepoint_array[0, sim_id,
                                                                   param_id]),
                                               param_id] -
                         nb_objects_all_states[1, int(transition_parameters[
                                                         transition_nb, 1]),
                                               sim_id, param_id])
                        / parameter_value_array[int(transition_parameters[
                                                        transition_nb, 1]),
                                                int(timepoint_array[0, sim_id,
                                                                    param_id]),
                                                param_id])

                # if (transition_nb == 13) & (nb_objects_all_states[1, transition_nb,
                #                                 sim_id, param_id] > 0):
                #     print(222,
                #           current_transition_rates[transition_nb,
                #                                    sim_id, param_id],
                #           parameter_value_array[int(transition_parameters[
                #                                         transition_nb, 1]),
                #                                 int(timepoint_array[0, sim_id,
                #                                                     param_id]),
                #                                 param_id],
                #           nb_objects_all_states[1, transition_nb,
                #                                 sim_id, param_id]
                #           )

        else:
            nb_objects = nb_objects_all_states[0, int(start_state),
                                               sim_id, param_id]
            current_transition_rates[transition_nb,
                                     sim_id, param_id] = (nb_objects *
                                                          transition_rate)

        current_rate = current_transition_rates[transition_nb, sim_id, param_id]

        if nb_parallel_cores[sim_id, param_id] > 1:
            cuda.atomic.add(total_rates, (sim_id, param_id), current_rate)
        else:
            total_rates[sim_id, param_id] += current_rate

        transition_nb += 1


    # if the rate of at least one parameter depends on the position,
    # don't use multicore across transitions but rather across objects
    if not position_dependence:
        return

    if nb_parallel_cores[sim_id, param_id] > 1:
        cuda.syncwarp(thread_masks[0,sim_id, param_id])

    # print(4444)
    # print(current_transition_rates[11, sim_id, param_id])

    # if (core_id == 0) & (sim_id == 0) & (param_id == 0):
    #     print(11, current_transition_rates[0, sim_id, param_id],
    #           current_transition_rates[1, sim_id, param_id],
    #           current_transition_rates[2, sim_id, param_id],
    #           current_transition_rates[3, sim_id, param_id],
    #           current_transition_rates[5, sim_id, param_id],
    #           total_rates[sim_id, param_id],
    #           nb_objects_all_states[0, 1,
    #                                 sim_id, param_id],
    #           nb_objects_all_states[0, 2,
    #                                 sim_id, param_id],
    #           nb_objects_all_states[0, 3,
    #                                 sim_id, param_id],
    #           nb_objects_all_states[0, 4,
    #                                 sim_id, param_id],
    #           nb_objects_all_states[0, 5,
    #                                 sim_id, param_id],
    #           total_density[0, sim_id, param_id],
    #           total_density[1, sim_id, param_id],
    #           total_density[2, sim_id, param_id],
    #           properties_array[0,1,0, sim_id, param_id],
    #           times[0,sim_id, param_id]
    #           )
    transition_nb = 0
    last_transition_nb = nb_transitions
    while transition_nb < last_transition_nb:
        transition_states = all_transition_states[transition_nb]
        # if the first value is nan then the parameter has no position
        # dependence
        if not math.isnan(params_prop_dependence[int(transition_parameters[
                                                        transition_nb, 0]), 2]):
            # if there is a position dependence, check whether the transition
            # actually might be executed (are there objects with the
            # the correct start state). Position dependent nucleation rate
            # is not supported at this stage and in fact would not need
            # per-object calculations.
            # print(current_transition_rates.shape[0])
            if (nb_objects_all_states[0, int(transition_states[0]),
                                      sim_id, param_id] > 0):
                _get_rate_of_prop_dependent_transition(params_prop_dependence[int(
                                                        transition_parameters[
                                                            transition_nb, 0])],
                                                      parameter_value_array,
                                                       transition_parameters,
                                                      object_dependent_rates,
                                                      transition_nb,
                                                      current_transition_rates,
                                                      total_rates,
                                                      transition_states[0],
                                                      object_states,
                                                      properties_array,
                                                      property_extreme_values,
                                                      timepoint_array,
                                                      first_last_idx_with_object,
                                                      nb_parallel_cores,
                                                      sim_id, param_id, core_id)
                # print(current_transition_rates[11, sim_id, param_id])
        transition_nb += 1

    if nb_parallel_cores[sim_id, param_id] > 1:
        cuda.syncwarp(thread_masks[0,sim_id, param_id])

    # if (core_id == 0) & (sim_id == 0) & (param_id == 0) & (times[0, sim_id, param_id] < 0.2):
    #     print(22, current_transition_rates[0, sim_id, param_id],
    #           current_transition_rates[1, sim_id, param_id],
    #           current_transition_rates[2, sim_id, param_id],
    #           current_transition_rates[3, sim_id, param_id],
    #           current_transition_rates[5, sim_id, param_id],
    #           total_rates[sim_id, param_id],
    #           nb_objects_all_states[0, 1,
    #                                 sim_id, param_id],
    #           nb_objects_all_states[0, 2,
    #                                 sim_id, param_id],
    #           nb_objects_all_states[0, 3,
    #                                 sim_id, param_id],
    #           nb_objects_all_states[0, 4,
    #                                 sim_id, param_id],
    #           nb_objects_all_states[0, 5,
    #                                 sim_id, param_id],
    #           total_density[0, sim_id, param_id],
    #           total_density[1, sim_id, param_id],
    #           total_density[2, sim_id, param_id],
    #           properties_array[0,0,0, sim_id, param_id],
    #           properties_array[0,1,0, sim_id, param_id],
    #           properties_array[0,2,0, sim_id, param_id],
    #           times[0,sim_id, param_id]
    #           )

def _get_rate_of_density_dependent_transition(params_prop_dependence,
                                          parameter_value_array,
                                          object_dependent_rates,
                                              local_density, local_resolution,
                                         transition_nb,
                                         current_transition_rates,
                                         total_rates,
                                         start_state,
                                         object_states,
                                         properties_array,
                                         property_extreme_values,
                                          timepoint_array,
                                         first_last_idx_with_object,
                                         nb_parallel_cores,
                                         sim_id, param_id, core_id):

    nb_objects = first_last_idx_with_object[1,sim_id, param_id] + 1
    (object_nb,
     last_object_pos) = _get_first_and_last_object_pos(nb_objects,
                                                    nb_parallel_cores[sim_id,
                                                                      param_id],
                                                    core_id)
    # max_position = property_extreme_values[1, 0, 0]

    while object_nb < last_object_pos:
        if object_states[0, object_nb, sim_id, param_id] == start_state:
            # get actual rate of the object depending on its position
            # if the start value of position dependence is defined,
            # calculate the change from the start position value
            # otherwise take the end value as the starting state

            if math.isnan(params_prop_dependence[0]):
                base_value = parameter_value_array[int(
                                       params_prop_dependence[1]),
                                                         int(timepoint_array[
                                                                 0, sim_id,
                                                                 param_id]),
                                                         param_id]
            else:
                base_value = parameter_value_array[int(
                                       params_prop_dependence[0]),
                                                         int(timepoint_array[
                                                                 0, sim_id,
                                                                 param_id]),
                                                         param_id]

            # base_value = cuda.selp(math.isnan(params_prop_dependence[0]),
            #                        parameter_value_array[int(
            #                            params_prop_dependence[1]),
            #                                              int(timepoint_array[
            #                                                      0, sim_id,
            #                                                      param_id]),
            #                                              param_id],
            #                        parameter_value_array[int(
            #                            params_prop_dependence[0]),
            #                                              int(timepoint_array[
            #                                                      0, sim_id,
            #                                                      param_id]),
            #                                              param_id])

            # add all defined properties to get the total property value
            # that the parameter value depends on
            end_position = 0
            dependence_prop_nb = 7
            while dependence_prop_nb < params_prop_dependence.shape[0]:
                if math.isnan(params_prop_dependence[dependence_prop_nb]):
                    # first nan indicates that there
                    # are more properties to consider
                    break
                end_position += properties_array[0,
                                                 int(params_prop_dependence[
                                                         dependence_prop_nb]),
                                                 object_nb,
                                                 sim_id, param_id]

                dependence_prop_nb += 1

            end_position = math.floor(end_position / local_resolution)

            density_ratio = (local_density[0, end_position, sim_id, param_id] /
                             local_density[1, end_position, sim_id, param_id])

            final_rate = base_value * density_ratio

            # # if (transition_nb == 4) | (transition_nb == 10):
            # print(final_rate, rate_diff, base_value,
            #       end_position, max_position, position_diff,
            #       # params_prop_dependence[6], # 0 for linear
            #       # params_prop_dependence[2], # 0 for absolute change
            #       # params_prop_dependence[4] # change value, nan for none defined
            #       )

            object_dependent_rates[int(params_prop_dependence[5]),
                                   object_nb, sim_id, param_id] = final_rate

            cuda.atomic.add(current_transition_rates,
                            (transition_nb, sim_id, param_id),
                            final_rate)
            cuda.atomic.add(total_rates, (sim_id, param_id), final_rate)

        object_nb += 1

def _get_rate_of_prop_dependent_transition(params_prop_dependence,
                                          parameter_value_array,
                                           transition_parameters,
                                          object_dependent_rates,
                                         transition_nb,
                                         current_transition_rates,
                                         total_rates,
                                         start_state,
                                         object_states,
                                         properties_array,
                                         property_extreme_values,
                                          timepoint_array,
                                         first_last_idx_with_object,
                                         nb_parallel_cores,
                                         sim_id, param_id, core_id):

    nb_objects = first_last_idx_with_object[1,sim_id, param_id] + 1
    (object_nb,
     last_object_pos) = _get_first_and_last_object_pos(nb_objects,
                                                    nb_parallel_cores[sim_id,
                                                                      param_id],
                                                    core_id)
    max_position = parameter_value_array[int(property_extreme_values[1, 0, 0]),
                                         int(timepoint_array[
                                                 0, sim_id, param_id]),
                                         param_id]

    while object_nb < last_object_pos:
        if object_states[0, object_nb, sim_id, param_id] == start_state:
            # get actual rate of the object depending on its position
            # if the start value of position dependence is defined,
            # calculate the change from the start position value
            # otherwise take the end value as the starting state

            if math.isnan(params_prop_dependence[0]):
                base_value =  parameter_value_array[int(
                                       params_prop_dependence[1]),
                                                         int(timepoint_array[
                                                                 0, sim_id,
                                                                 param_id]),
                                                         param_id]
            else:
                base_value = parameter_value_array[int(
                                       params_prop_dependence[0]),
                                                         int(timepoint_array[
                                                                 0, sim_id,
                                                                 param_id]),
                                                         param_id]

            # base_value = cuda.selp(math.isnan(params_prop_dependence[0]),
            #                        parameter_value_array[int(
            #                            params_prop_dependence[1]),
            #                                              int(timepoint_array[
            #                                                      0, sim_id,
            #                                                      param_id]),
            #                                              param_id],
            #                        parameter_value_array[int(
            #                            params_prop_dependence[0]),
            #                                              int(timepoint_array[
            #                                                      0, sim_id,
            #                                                      param_id]),
            #                                              param_id])

            # add all defined properties to get the total property value
            # that the parameter value depends on
            end_position = 0
            dependence_prop_nb = 7
            while dependence_prop_nb < params_prop_dependence.shape[0]:
                if math.isnan(params_prop_dependence[dependence_prop_nb]):
                    # first nan indicates that there
                    # are more properties to consider
                    break
                end_position += properties_array[0,
                                                 int(params_prop_dependence[
                                                         dependence_prop_nb]),
                                                 object_nb,
                                                 sim_id, param_id]

                # print(object_states[0, object_nb, sim_id, param_id],
                #       params_prop_dependence[dependence_prop_nb],
                #       int(params_prop_dependence[dependence_prop_nb]),
                #           dependence_prop_nb,
                #           properties_array[0,
                #                                  int(params_prop_dependence[
                #                                          dependence_prop_nb]),
                #                                  object_nb,
                #                                  sim_id, param_id])
                dependence_prop_nb += 1

            if math.isnan(params_prop_dependence[0]):
                position_diff = max_position - end_position
            else:
                position_diff = end_position
            # position_diff = cuda.selp(math.isnan(params_prop_dependence[0]),
            #                           max_position - end_position,
            #                           end_position)

            # if the change is per absolute position diff (0) or relative
            # position difference (1)
            if params_prop_dependence[3] != 0:
                position_diff = position_diff/max_position
            # position_diff = cuda.selp(params_prop_dependence[3] == 0,
            #                           position_diff,
            #                           position_diff/max_position)

            # position_diff = min(position_diff, 10)

            # if change is not defined, get it from difference between
            # end and start value, divided by either absolute length in um
            # or relative length (1, unchanged value)
            if math.isnan(params_prop_dependence[4]):
                rate_diff = (parameter_value_array[int(
                                     params_prop_dependence[1]),
                                                        int(timepoint_array[
                                                                0, sim_id,
                                                                param_id]),
                                                        param_id] -
                                  parameter_value_array[int(
                                      params_prop_dependence[0]),
                                                        int(timepoint_array[
                                                                0, sim_id,
                                                                param_id]),
                                                        param_id])
                if params_prop_dependence[3] == 0:
                    rate_diff /= max_position
            else:
                rate_diff = parameter_value_array[int(
                                      params_prop_dependence[4]),
                                                        int(timepoint_array[
                                                                0, sim_id,
                                                                param_id]),
                                                        param_id]
            # rate_diff = cuda.selp(math.isnan(params_prop_dependence[4]),
            #
            #                      (parameter_value_array[int(
            #                          params_prop_dependence[1]),
            #                                             int(timepoint_array[
            #                                                     0, sim_id,
            #                                                     param_id]),
            #                                             param_id] -
            #                       parameter_value_array[int(
            #                           params_prop_dependence[0]),
            #                                             int(timepoint_array[
            #                                                     0, sim_id,
            #                                                     param_id]),
            #                                             param_id])
            #                       / cuda.selp(params_prop_dependence[3] == 0,
            #                                   max_position, float(1)),
            #
            #                       parameter_value_array[int(
            #                           params_prop_dependence[4]),
            #                                             int(timepoint_array[
            #                                                     0, sim_id,
            #                                                     param_id]),
            #                                             param_id])

            if params_prop_dependence[6] == 0:
                if params_prop_dependence[2] == 0:
                    rate_diff *= position_diff
                else:
                    rate_diff *= base_value * position_diff
            else:
                rate_diff = base_value * math.exp(- rate_diff * position_diff)
            # 0 for linear, 1 for exponential property dependence
            # rate_diff = cuda.selp(params_prop_dependence[6] == 0,
            #
            #                       cuda.selp(params_prop_dependence[2] == 0,
            #                                 rate_diff * position_diff,
            #                                 rate_diff * base_value
            #                                 * position_diff),
            #
            #                       base_value *
            #                       math.exp(- rate_diff * position_diff))

            # for linear parameter dependence:
            # Prevent changes through parameter dependence smaller than 0
            # which would mean a reduction of the separately defined baseline
            # value. Such a reduction should be implemented through a reduced
            # separately defined baseline value of the parameter and a steeper
            # change of the position dependence.
            # For the case where a rate should be constant in most position
            # but then reduced at the beginning or end, a reduction through a
            # separate value should still be possible. In that case the
            # mininmum allowed value is the negative rate of the transition
            # (for one object). In that case the base_value has to be < 0.
            # The maximum final rate allowed is the maximum of the value in the
            # start and end, don't allow values at positions in between the
            # start and the end to be higher than the maximum of both.

            if params_prop_dependence[6] == 0:
                if base_value < 0:
                    final_rate = max(- parameter_value_array[int(
                        transition_parameters[transition_nb, 0]),
                                                int(
                                                    timepoint_array[
                                                        0, sim_id, param_id]),
                                                param_id],

                        max(min(parameter_value_array[int(
                            params_prop_dependence[0]), int(
                            timepoint_array[0, sim_id, param_id]),
                                                      param_id],
                                parameter_value_array[int(
                                    params_prop_dependence[
                                        1]),
                                                      int(
                                                          timepoint_array[
                                                              0, sim_id,
                                                              param_id]),
                                                      param_id], ),
                            min(0, base_value + rate_diff)))
                else:
                    final_rate = min(max(parameter_value_array[int(
                                      params_prop_dependence[0]), int(
                                      timepoint_array[0, sim_id, param_id]),
                                                                param_id],

                                          parameter_value_array[int(
                                              params_prop_dependence[1]),
                                                          int(timepoint_array[
                                                                  0, sim_id,
                                                                  param_id]),
                                                                param_id],),

                                      max(0, base_value + rate_diff))
            else:
                final_rate = rate_diff

            # final_rate = cuda.selp(params_prop_dependence[6] == 0,
            #
            #                         cuda.selp(base_value < 0,
            #
            #                         max(- parameter_value_array[int(
            #                             transition_parameters[transition_nb,0]),
            #                                                     int(
            #                             timepoint_array[0, sim_id,param_id]),
            #                                                     param_id],
            #
            #                             max(min(parameter_value_array[int(
            #                                 params_prop_dependence[0]),int(
            #                                 timepoint_array[0, sim_id,param_id]),
            #                                                           param_id],
            #                                           parameter_value_array[int(
            #                                               params_prop_dependence[
            #                                                   1]),
            #                                                 int(
            #                                                   timepoint_array[
            #                                                       0, sim_id,
            #                                                       param_id]),
            #                                               param_id],),
            #                                 min(0, base_value + rate_diff))),
            #
            #                       min(max(parameter_value_array[int(
            #                           params_prop_dependence[0]), int(
            #                           timepoint_array[0, sim_id, param_id]),
            #                                                     param_id],
            #
            #                               parameter_value_array[int(
            #                                   params_prop_dependence[1]),
            #                                               int(timepoint_array[
            #                                                       0, sim_id,
            #                                                       param_id]),
            #                                                     param_id],),
            #
            #                           max(0, base_value + rate_diff))
            #                         ),
            #
            #                        rate_diff)


            # final_rate = cuda.selp(params_prop_dependence[6] == 0,
            #
            #                                   min(max(params_prop_dependence[0],
            #                                           params_prop_dependence[1]),
            #                                       max(0,
            #                                           base_value + rate_diff)),
            #
            #                                 rate_diff)

            # if (transition_nb == 13) & (final_rate != 0):
            #     print(final_rate, rate_diff, base_value,
            #           end_position, max_position, position_diff,
            #           params_prop_dependence[0], params_prop_dependence[1],
            #           - parameter_value_array[int(transition_parameters[
            #                                           transition_nb, 0]),
            #                                   int(timepoint_array[0, sim_id,
            #                                                       param_id]),
            #                                   param_id],
            #           parameter_value_array[int(
            #               params_prop_dependence[4]),
            #                                 int(timepoint_array[
            #                                         0, sim_id,
            #                                         param_id]),
            #                                 param_id]
            #           # params_prop_dependence[6], # 0 for linear
            #           # params_prop_dependence[2], # 0 for absolute change
            #           # params_prop_dependence[4] # change value, nan for none defined
            #           )

            object_dependent_rates[int(params_prop_dependence[5]),
                                   object_nb, sim_id, param_id] = final_rate

            cuda.atomic.add(current_transition_rates,
                            (transition_nb, sim_id, param_id),
                            final_rate)
            cuda.atomic.add(total_rates, (sim_id, param_id), final_rate)

        object_nb += 1



def _get_tmin_tmax_for_property_changes(property_changes_tminmax_array,
                                       property_changes_per_state,
                                       total_property_changes,
                                        parameter_value_array,
                                        timepoint_array,
                                        property_extreme_values,
                                       current_sum_tmax,

                                       properties_tmax,
                                       properties_tmax_array,
                                       end_position_tmax_array,

                                       first_last_idx_with_object,
                                       object_states, properties_array ,

                                        nb_parallel_cores,  core_id,
                                       sim_id, param_id
                                       ):

    nb_objects = first_last_idx_with_object[1,sim_id, param_id] + 1
    (object_nb,
     last_object_pos) = _get_first_and_last_object_pos(nb_objects,
                                                    nb_parallel_cores[sim_id, param_id],
                                                    core_id)
    while object_nb < last_object_pos:
        # check if action should be executed on object
        state = int(object_states[0, object_nb, sim_id, param_id])
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
                    tmax = - (properties_array[0, net_change_property_nb,
                                            object_nb,
                                            sim_id, param_id] / net_change)

                    properties_tmax[0, core_id, net_change_property_nb-1,
                                    sim_id, param_id] = tmax
                net_change_property_nb += 1
            # sort indices of properties by size of tmax (smaller to larger)

            nb_sorted_tmax = 0
            if not math.isnan(properties_tmax[0, core_id, 0, sim_id, param_id]):
                properties_tmax[1, core_id, 0, sim_id, param_id] = 0
                nb_sorted_tmax += 1

            property_idx = 1
            while property_idx < properties_tmax.shape[1]:
                tmax = properties_tmax[0, core_id, property_idx, sim_id, param_id]
                if not math.isnan(tmax):
                    nb_sorted_tmax += 1
                    # go through already sorted tmax indices and check which idx
                    # is current tmax smaller than
                    property_idx_inner = 0
                    while property_idx_inner < nb_sorted_tmax:
                        tmax_inner = properties_tmax[1, core_id,
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
                                property_idx_sort = property_idx_new_sort
                                sorted_tmax = properties_tmax[1, core_id,
                                                              property_idx_sort
                                                              - 1,
                                                              sim_id, param_id]
                                properties_tmax[1, core_id,
                                                property_idx_new_sort,
                                                sim_id,
                                                param_id] = sorted_tmax
                                property_idx_new_sort -= 1

                            properties_tmax[1, core_id,
                                            property_idx_inner,
                                            sim_id, param_id] = property_idx
                            break
                        # if it was not smaller than any previously sorted tmax
                        # add it at the end
                        if property_idx_inner == property_idx:
                            properties_tmax[1, core_id, property_idx,
                                            sim_id, param_id] = property_idx
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
                property_val = properties_array[0, property_tmax_nb,
                                                object_nb,
                                                sim_id, param_id]
                properties_tmax_array[core_id, 0, property_tmax_nb,
                                      sim_id, param_id] = property_val
                property_tmax_nb += 1

            first_position = properties_array[0, 0, object_nb,
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
                property_idx = int(properties_tmax[1, core_id, tmax_idx,
                                                   sim_id, param_id])
                tmax = properties_tmax[0, core_id, property_idx,
                                       sim_id, param_id]
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
                    position_diff = end_position - parameter_value_array[int(
                        property_extreme_values[1, 0, 0]),
                        int(timepoint_array[0, sim_id, param_id]), param_id]
                    tmax_object_removal = (position_diff /
                                           current_sum_tmax[core_id,
                                                            tmax_idx,
                                                            sim_id,
                                                            param_id])
                    break

                # get time at which end_position reaches the max position
                # don't allow the tmax_end_position to be overwritten once
                # defined
                if ((end_position > parameter_value_array[int(
                        property_extreme_values[1, 0, 0]),
                        int(timepoint_array[0, sim_id, param_id]), param_id]) &
                        (math.isnan(tmax_end_position))):
                    if current_sum_tmax[core_id,tmax_idx, sim_id, param_id] > 0:
                        # the difference from the current tmax is
                        # the difference of the end_position minus the
                        # max position, divided by the current sum
                        position_diff = end_position - parameter_value_array[
                            int(property_extreme_values[1, 0, 0]),
                            int(timepoint_array[0, sim_id, param_id]), param_id]
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
                        position_diff = (parameter_value_array[int(
                            property_extreme_values[1, 0, 0]),
                            int(timepoint_array[0, sim_id, param_id]), param_id]
                                         - end_position)
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
                            position = properties_array[0, 0, object_nb,
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
                        property_tmax = properties_tmax[0, core_id,
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

                    property_changes_tminmax_array[0, object_nb,
                                                net_change_property_nb,
                                                sim_id, param_id] = tmin

                    property_changes_tminmax_array[1, object_nb,
                                                net_change_property_nb,
                                                sim_id, param_id] = tmax

                net_change_property_nb += 1
        object_nb += 1


def _get_tau(total_rates, first_last_idx_with_object, object_states,
            property_changes_per_state,nucleation_changes_per_state,
            property_changes_tminmax_array, eq_terms,
             rng_states, simulation_factor, parameter_factor,
             nb_parallel_cores,  thread_masks, core_id, sim_id, param_id):

        rate_baseline = total_rates[sim_id, param_id]
        random_nb = _get_random_number(sim_id, param_id, rng_states,
                                       simulation_factor, parameter_factor)
        rand_exp = - (math.log(random_nb))

        tau_guess = rand_exp/rate_baseline
        first_order = rate_baseline

        eq_terms[1, 0, sim_id, param_id] = 0
        eq_terms[0, 0, sim_id, param_id] = - rand_exp

        tau_error = 1
        lowest_error = 1
        best_tau = 0
        nb = 0
        while tau_error > 0.005:
            eq_terms[1, 1, sim_id, param_id] = 0
            eq_terms[0, 1, sim_id, param_id] = 0

            if nb == 0:
                calculate_second_order_base = True
            else:
                calculate_second_order_base = False

            nb_objects = first_last_idx_with_object[1,sim_id, param_id] + 1
            (object_nb,
             last_object_nb) = _get_first_and_last_object_pos(nb_objects,
                                                              nb_parallel_cores[sim_id, param_id],
                                                              core_id)
            while object_nb < last_object_nb:
                state = object_states[0, object_nb, sim_id, param_id]

                if state > 0:
                    property_idx = 0
                    while property_idx < property_changes_per_state.shape[1]:
                        net_change = nucleation_changes_per_state[state-1,
                                                                property_idx,
                                                                sim_id,
                                                                param_id]
                        if net_change != 0:
                            tmin = property_changes_tminmax_array[0,object_nb,
                                                               property_idx,
                                                               sim_id, param_id]
                            tmax = property_changes_tminmax_array[1,object_nb,
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
                                        cuda.atomic.add(eq_terms,
                                                        (1, 0, sim_id, param_id),
                                                        net_change)
                                        cuda.atomic.add(eq_terms,
                                                        (0, 0, sim_id, param_id),
                                                        - (math.pow(tmin,2)/2))
                                    elif tmax > 0:
                                        if tau_guess < tmax:
                                            # if tau is before tmax, add net_change
                                            # to variable for tau dependent change
                                            # (second_order)
                                            cuda.atomic.add(eq_terms,
                                                            (1, 1, sim_id, param_id),
                                                            net_change)
                                            cuda.atomic.add(eq_terms,
                                                            (0, 1, sim_id, param_id),
                                                            - net_change *
                                                            (math.pow(tmin,2)) / 2)

                                        else:
                                            # if tau is smaller then tmax, add
                                            # net_change to variable for tau independent
                                            # change (eq_terms[0] for constant
                                            # term)
                                            cuda.atomic.add(eq_terms,
                                                            (0, 1, sim_id,
                                                             param_id),
                                                            (net_change *
                                                             (math.pow(tmax,2) -
                                                              math.pow(tmin,2)))
                                                            / 2)

                        property_idx += 1
                object_nb += 1

            if nb_parallel_cores[sim_id, param_id] > 1:
               cuda.syncwarp(thread_masks[0,sim_id, param_id])

            total_constant = (eq_terms[0, 0, sim_id, param_id] +
                              eq_terms[0, 1, sim_id, param_id])

            # divide by two due to integral
            total_second_order = (eq_terms[1, 0, sim_id, param_id] +
                                  eq_terms[1, 1, sim_id, param_id])/2

            if total_second_order != 0:
                # for the case where the sum under the root is negative,
                # use complex math and take the real part.
                # This can happen more or less often depending on the specific
                # model. This happens particularly if there is an overall
                # negative net change for all the objects present
                # and a relatively large tau_guess. Once more objects are
                # present, a smaller tau makes a negative root much less likely.
                new_tau_guess_sqrt = cmath.sqrt(math.pow(first_order,2) -
                                               4*total_second_order*
                                               total_constant).real
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
            if nb == 10:
                tau_guess = best_tau
                break

            tau_guess = tau_guess + (new_tau_guess - tau_guess) * 0.4
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
    # print(1234, total_rates[sim_id, param_id])
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
        if round(current_rate_sum, 6) >= round(threshold, 6):
            current_transitions[sim_id, param_id] = transition_nb
            return
        transition_nb += 1

    print(666666, threshold, current_rate_sum, total_rates[sim_id, param_id],
          random_number)

    return


def _determine_positions_of_transitions(current_transitions,
                                        all_transition_states,
                                        nb_objects_all_states,
                                        all_transition_positions,
                                            params_prop_dependence,
                                            object_dependent_rates,
                                        parameter_value_array,
                                        transition_parameters,
                                        current_transition_rates,
                                        timepoint_array,
                                        object_states, core_id,
                                        sim_id, param_id, rng_states,
                                        first_last_idx_with_object,
                                        simulation_factor, parameter_factor):

    # the transitions masks only tell which reaction happens in each
    # stimulation, but not which object in this simulation is affected
    # To get one random object of the possible objects,
    # first, create mask of index positions, so that each object for each
    # simulation has a unique identifier (index) within this simulation
    # setting all positions where no catastrophe can take place to 0
    all_transition_positions[sim_id, param_id] = math.nan
    if math.isnan(current_transitions[sim_id, param_id]):
        print(55555)
        return

    start_state = all_transition_states[int(current_transitions[sim_id,
                                                                param_id]), 0]

    # if start state is 0, choose the first object with state 0
    if start_state == 0:
        # if the possible lowest idx with no object actually has an object now
        # increase index until an index without object is found and save this
        # position
        if core_id == 0:
            if object_states[0, first_last_idx_with_object[0,sim_id, param_id],
                             sim_id, param_id] != 0:
                _increase_lowest_no_object_idx(first_last_idx_with_object,
                                               object_states, sim_id, param_id)
            all_transition_positions[sim_id,
                                     param_id] = first_last_idx_with_object[0,
                                                                            sim_id,
                                                                            param_id]
        return
    else:
        # if the highest idx with object vanished
        # search for the new highest object idx
        if object_states[0, first_last_idx_with_object[1, sim_id, param_id],
                         sim_id, param_id] == 0:
            _reduce_highest_object_idx(first_last_idx_with_object,
                                       object_states, sim_id, param_id)

        # if the current transition does not have a dependence, choose a random
        # object for the transition
        # params_prop_dependence needs the parameter number which one gets
        # from transition parameters at the idx of the transition number
        if math.isnan(params_prop_dependence[
                          int(transition_parameters[int(
                              current_transitions[sim_id, param_id]
                          ), 0]), 2]):
            # for all other states, choose a random position with that state
            nb_objects = nb_objects_all_states[0, int(start_state),
                                               sim_id, param_id]
            random_object_pos = math.floor(_get_random_number(sim_id, param_id,
                                                         rng_states,
                                                         simulation_factor,
                                                         parameter_factor)
                                           * nb_objects)
            # allow a maximum of nb_objects - 1
            random_object_pos = int(min(nb_objects-1, random_object_pos))
            object_pos = 0
            current_nb_state_objects = 0

            # go through all objects, check which one is in the start_state
            # and then choose the nth (n=random_object_pos) object that is in
            # the start_state
            while object_pos <= first_last_idx_with_object[1,sim_id, param_id]:# object_states.shape[1]:#
                # if object_state > 4:
                #     print(555, object_state, object_pos)
                if (object_states[0, object_pos, sim_id, param_id] ==
                        start_state):

                    if current_nb_state_objects == random_object_pos:
                        all_transition_positions[sim_id, param_id] = object_pos
                        return
                    current_nb_state_objects += 1
                object_pos += 1

            if core_id == 0:
                print(33333, object_pos, first_last_idx_with_object[1,sim_id, param_id],
                      current_nb_state_objects, random_object_pos,
                      nb_objects,
                      start_state,
                      sim_id, param_id
                      # int(current_transitions[sim_id,
                      #                          param_id]),
                      # all_transition_states.shape[0]
                      )
                print(object_states[0, 0, sim_id, param_id],
                      object_states[0, 1, sim_id, param_id],
                      object_states[0, 2, sim_id, param_id],
                      object_states[0, 3, sim_id, param_id],
                      object_states[0, 4, sim_id, param_id],
                      object_states[0, 5, sim_id, param_id],
                      object_states[0, 6, sim_id, param_id],
                      object_states[0, 7, sim_id, param_id],
                      object_states[0, 8, sim_id, param_id],
                      object_states[0, 9, sim_id, param_id],
                      object_states[0, 10, sim_id, param_id],
                      object_states[0, 11, sim_id, param_id],
                      object_states[0, 12, sim_id, param_id],
                      object_states[0, 13, sim_id, param_id],
                      object_states[0, 14, sim_id, param_id],
                      object_states[0, 15, sim_id, param_id],
                      object_states[0, 16, sim_id, param_id],
                      object_states[0, 17, sim_id, param_id],
                      object_states[0, 18, sim_id, param_id],
                      object_states[0, 19, sim_id, param_id],
                      object_states[0, 20, sim_id, param_id],
                      object_states[0, 21, sim_id, param_id])
        else:
            # print(nb_objects_all_states[1, sim_id, param_id],
            #       nb_objects_all_states[2, sim_id, param_id],
            #       nb_objects_all_states[3, sim_id, param_id],
            #       nb_objects_all_states[4, sim_id, param_id])
            # if the current transition does have a dependence, then choose
            # an object for the transition proportional to its dependent rate
            transition_rate = current_transition_rates[int(current_transitions[
                                                               sim_id, param_id]
                                                           ),
                                                       sim_id, param_id]
            # the random threshold is the total transition rate for the
            # current ransition multiplied by a uniform random number in the
            # interval [0, 1]
            random_thresh = (_get_random_number(sim_id, param_id,
                                                         rng_states,
                                                         simulation_factor,
                                                         parameter_factor)
                             * transition_rate)

            baseline_rate = parameter_value_array[int(
                transition_parameters[int(current_transitions[sim_id, param_id]
                                          ), 0]),
                                                  int(timepoint_array[0, sim_id,
                                                                      param_id]),
                                                  param_id]
            dependence_idx = int(params_prop_dependence[
                                     int(transition_parameters[
                                             int(current_transitions[
                                                     sim_id, param_id]), 0]), 5])

            # accumulate the linear and baseline rates for each object
            # until the threshold is crossed, then set the transition position
            # as the object position that crossed the threshold
            current_sum = 0
            object_pos = 0
            # if current_transitions[sim_id, param_id] != 5:
            #     if current_transitions[sim_id, param_id] != 11:
            #         print(current_transitions[sim_id, param_id])

            # if object_states[0, object_pos, sim_id, param_id] != 2:
            #     if object_states[0, object_pos, sim_id, param_id] != 5:
            #         print(object_states[0, object_pos, sim_id, param_id])
            # print(1111)
            while object_pos <= first_last_idx_with_object[1,sim_id, param_id]:
                if (object_states[0, object_pos, sim_id, param_id] ==
                        start_state):

                    # print(2222)
                    # print(object_dependent_rates[dependence_idx,
                    #                                       object_pos,
                    #                                       sim_id, param_id])
                    current_sum += baseline_rate
                    current_sum += object_dependent_rates[dependence_idx,
                                                          object_pos,
                                                          sim_id, param_id]

                    if round(current_sum, 6) >= round(random_thresh, 6):
                        all_transition_positions[sim_id, param_id] = object_pos
                        return
                object_pos += 1

            print(44444, object_pos, current_sum, random_thresh,
                  transition_rate,
                  baseline_rate, start_state,
                  nb_objects_all_states[0, int(start_state), sim_id, param_id])

    return


def _track_object_property_changes(object_nb, property_nb, new_value,
                                   properties_array,
                                   local_object_lifetime_array,
                                   local_lifetime_resolution,
                                   times, sim_id, param_id):
    """
    Whenever a property is changed, a function with the property number, the
    previous property value and the new property value should be executed.
    For multiple property values changing, first the property closest
    to the tip has to be changed (highest property number).

    Lifetime tracking does not work well if some populations move retrogradely
    and some anterogradely
	If an object would move out of the system at one open end, the lifetime
	information there might be overwritten (if the length of the object is
	larger than the system length). When the object (once transitioned to a
	different population) would move in again, the lifetime information is lost
	already and cant be recovered - leaving a section of the object in the
	system without a lifetime.
    """
    # check if lifetime should be tracked by checking that the length of
    # the first and second index is not 1 (which would mean only one MT and
    # only one length segment allowed, but is actually used as a placeholder
    # when lifetime should not be tracked)
    if ((local_object_lifetime_array.shape[0] == 1) &
            (local_object_lifetime_array.shape[1] == 1)):
        return
    # if the changed property is the position (property_nb == 0), no lifetime
    # change needs to be tracked since the actual MT is not changing
    if property_nb == 0:
        return

    # get start point for current property change
    # first add all properties before the current property_nb
    base = 0
    # start adding from property_nb 1, since the position does not matter
    # for the timestamp
    start_property_nb = 1
    while start_property_nb < property_nb:
        if not math.isnan(properties_array[0, start_property_nb, object_nb,
                                           sim_id, param_id]):
            base += properties_array[0, start_property_nb, object_nb, sim_id,
                                      param_id]
        start_property_nb += 1

    # then add the old property value
    start = base
    if not math.isnan(properties_array[0, int(property_nb),
                                       int(object_nb), sim_id, param_id]):
        start += properties_array[0, int(property_nb),
                                    int(object_nb), sim_id, param_id]

    # the end point is the base value plus the new value
    end = base + new_value

    # the first index in local_object_lifetime_array is nan, indicating the
    # start of the object (this is relevant, once the object length exceeds
    # the system length through moving out of an open end of the system

    # add one, since one idx is used as a marker for the beginning of the object
    # (with a 0 value)
    start_idx = math.floor(start / local_lifetime_resolution)
    if end == 0:
        end_idx = 0
    else:
        end_idx = math.floor(end / local_lifetime_resolution) + 1

    # if (start_idx != end_idx) & (object_nb == 0):
    #     print(start_idx, end_idx,
    #           properties_array[0, int(property_nb),
    #                            int(object_nb), sim_id, param_id],
    #           new_value
    #           )

    # change timestamp, if object moved out of an index (for both directions)
    # if end is larger than start, add timestamps
    if end_idx > start_idx:
        # only start filling after the start index, but until the end_idx
        start_idx += 1
        start_from_first_idx = True
        while start_idx <= end_idx:
            # if the current idx is larger than the size of the lifetime array
            # start again from the beginning of the lifetime array, by
            # subtracting the highest full multiple of the lifetime array size
            # that still keeps the index positive
            if start_idx >= local_object_lifetime_array.shape[1]:
                # Whether the MT actually starts at the first index,
                # or whether it starts at a later index (if it is longer than
                # the system but its startpoint is outside of the system)
                # THIS DOES NOT WORK IF THE ENDPOINT CAN BE OUTSIDE OF THE
                # SYSTEM
                start_from_first_idx = False
                # print(11, start_idx, end_idx,
                #       properties_array[0, int(property_nb),
                #                        int(object_nb), sim_id, param_id],
                #       new_value
                #       )
                # by subtracting both from start_idx and end_idx,
                # this calculation only needs to be done at the transition
                # beyond the size of the array
                end_idx -= int(math.floor(start_idx /
                                            local_object_lifetime_array.shape[1]
                                            ) *
                               local_object_lifetime_array.shape[1])
                start_idx -= int(math.floor(start_idx /
                                            local_object_lifetime_array.shape[1]
                                            ) *
                                            local_object_lifetime_array.shape[1]
                                 )

            local_object_lifetime_array[object_nb, int(start_idx),
                                        sim_id,
                                        param_id] = times[0, sim_id, param_id]
            start_idx += 1

        # if the MT does not start from the first idx anymore,
        # put the new start point one position after the end_idx
        if (not start_from_first_idx):
            # if the last index is reached, add the marker for the
            # beginning of the part of the MT in the neurite again (0 value)
            # if start_idx + 1 is larger than the lifetime array size
            # add the separator at the beginning of the array (idx 0)
            if int(end_idx + 1) == local_object_lifetime_array.shape[1]:
                local_object_lifetime_array[object_nb, 0,
                                            sim_id, param_id] = 0
            else:
                local_object_lifetime_array[object_nb, int(end_idx) + 1,
                                            sim_id, param_id] = 0
    else:
        # if end is smaller than start, remove timestamps
         # start removing at the start_index, but don't include the end_idx
         # (since it is still in the end_idx
        start_from_first_idx = True
        while start_idx > end_idx:
            # if the current idx is larger than the size of the lifetime array
            # start again from the beginning of the lifetime array, by
            # subtracting the highest full multiple of the lifetime array size
            # that still keeps the index positive
            if start_idx >= local_object_lifetime_array.shape[1]:
                start_from_first_idx = False
                start_idx_tmp = start_idx - int(math.floor(start_idx /
                                            local_object_lifetime_array.shape[1]
                                            ) *
                                 local_object_lifetime_array.shape[1])
                # print(22, start_idx, end_idx,
                #       properties_array[0, int(property_nb),
                #                        int(object_nb), sim_id, param_id],
                #       new_value
                #       )
            else:
                start_idx_tmp = start_idx
            local_object_lifetime_array[object_nb, int(start_idx_tmp),
                                        sim_id, param_id] = np.nan
            start_idx -= 1
        # move the startpoint back
        if (not start_from_first_idx) & (end_idx != 0):
            local_object_lifetime_array[object_nb, int(end_idx) + 1,
                                        sim_id, param_id] = 0



def _execute_actions_on_objects(parameter_value_array, action_parameters,
                                action_state_array,
                                properties_array,
                                nb_objects_all_states,
                                property_extreme_values,

                                local_object_lifetime_array,
                                                 local_lifetime_resolution,
                                local_resolution,

                                all_action_properties,
                                action_operation_array,
                                object_states,
                                reaction_time,
                                timepoint_array,
                                first_last_idx_with_object,
                                nb_parallel_cores,  core_id,
                                sim_id, param_id,
                                times):
    # execute actions on objects depending on state, before changing state
    action_nb = 0


    while action_nb < action_parameters.shape[0]:
        action_parameter = int(action_parameters[action_nb])
        # get the current action value, dependent on reaction time
        action_value = parameter_value_array[action_parameter,
                                             int(timepoint_array[
                                                     0, sim_id, param_id]),
                                             param_id]
        if action_value != 0:
            current_action_value = action_value * reaction_time
            action_states = action_state_array[action_nb]
            action_properties = all_action_properties[action_nb]
            # action operation is -1 for subtracting and 1 for adding
            action_operation = action_operation_array[action_nb]
            # go through each object and check whether its state is in
            # action_states,

            nb_objects = first_last_idx_with_object[1,sim_id, param_id] + 1
            (object_pos,
             last_object_pos) = _get_first_and_last_object_pos(nb_objects,
                                                            nb_parallel_cores[sim_id, param_id],
                                                            core_id)
            while object_pos < last_object_pos:
                object_state = object_states[0, object_pos, sim_id, param_id]
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
                    property_val = properties_array[0, int(property_nb),
                                                     object_pos,
                                                     sim_id, param_id]
                    new_property_val = (property_val +
                                        action_operation * current_action_value)

                    # if (sim_id == 0) & (param_id == 0) & (core_id == 0) & (times[0, sim_id, param_id] < 0.2):
                    #     print(33, property_nb, new_property_val, property_val,
                    #           action_operation, current_action_value)


                    # if (sim_id == 0) & (param_id == 0) & (core_id == 0) & (times[0, sim_id, param_id] < 0.2):
                    #     print(44, property_nb, new_property_val, property_val,
                    #           threshold, min_value[0],
                    #           len(min_value), min_value[1], min_value[2])

                    # check whether the property value is above the max val
                    # similarly as for the min val condition
                    max_value = property_extreme_values[1, int(property_nb)]
                    if math.isnan(max_value[0]):
                        threshold = math.nan
                    else:
                        threshold = parameter_value_array[int(max_value[0]),
                                                          int(timepoint_array[
                                                            0, sim_id,
                                                            param_id]),
                                                          param_id]


                    # the first index is the actual threshold value in any case
                    # if the second index is not nan, then follows the operation
                    # (whether other values should be added (1) or subtracted
                    # (-1)
                    # and then follow all the property value numbers that should
                    # be considered (e.g. if the current property nb is 1 then
                    # this is the property that the new_property_val corresponds
                    # to. If the index 2 and 3 then are 0 and 2 then these two
                    # property values should also be considered for the
                    # threshold. For operation == 1, this means that the sum of
                    # property 0, 1 and 3 should not go above the threshold
                    # this is echecked by subtracting (for operation = 1) the
                    # additional properties from the threshold, to obtain the
                    # actual maximum value (threshold) for the current property.
                    if max_value[1] == 0:
                        # 0 at index 1 means the threshold is open
                        # (see more detailed description above for min
                        #  threshold)
                        threshold = math.nan
                        # for properties with ObjectGeometry max values but
                        # open max (closed_max = False)
                        # the threshold (first index for that property) will be
                        # set to zero to indicate that there is no max
                        # value
                    elif len(max_value) == 2:
                        threshold = threshold
                    elif math.isnan(max_value[2]):
                        threshold = threshold
                    else:
                        max_value_nb = 3
                        while max_value_nb < len(max_value):
                            val_property_nb = max_value[max_value_nb]
                            if math.isnan(val_property_nb):
                                break
                            if max_value[2] == 1:
                                threshold = threshold - properties_array[0, int(val_property_nb),
                                                                         object_pos,
                                                                         sim_id,
                                                                         param_id]
                            if max_value[2] == -1:
                                threshold = threshold + properties_array[0, int(val_property_nb),
                                                                         object_pos,
                                                                         sim_id,
                                                                         param_id]
                            max_value_nb += 1
                    #
                    # if (sim_id == 356) & (object_pos == 4):
                    #     print(333, threshold, new_property_val,
                    #           len(max_value), max_value[0], max_value[1],
                    #           max_value[2], max_value[3])
                    if new_property_val >= threshold:
                        new_property_val = threshold

                        # # HARD CODE CATASTROPHE UPON REACHING MAX VALUE
                        # object_states[0, object_pos,
                        #                  sim_id, param_id] = 2
                        #
                        # cuda.atomic.add(nb_objects_all_states,
                        #                 (0, 1, sim_id, param_id),
                        #                 -1)
                        #
                        # cuda.atomic.add(nb_objects_all_states,
                        #                 (0, 2, sim_id, param_id),
                        #                 1)
                        #
                        # shrinkage = (new_property_val - threshold) / 9 * 15
                        #
                        # new_property_val = threshold
                        # new_property_val -= shrinkage
                        # new_property_val = max(0, new_property_val)

                    else:
                        # the property value cannot be higher than the max
                        # and higher than the min at the same time

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
                        min_value = property_extreme_values[0, int(property_nb)]
                        if math.isnan(min_value[0]):
                            threshold = math.nan
                        else:
                            threshold = parameter_value_array[int(min_value[0]),
                                                              int(timepoint_array[
                                                                0, sim_id, param_id]),
                                                              param_id]

                        # if the second index is 0, then the threshold is not closed
                        # therefore it just defines geometry and is not enforced
                        if min_value[1] == 0:
                            threshold = math.nan
                        elif len(min_value) == 2:
                            # this case would mean that the threshold is closed
                            # but that the extreme properties do not depend on other
                            # properties (therefore min_value is not longer than 2)
                            threshold = threshold
                        elif math.isnan(min_value[2]):
                            # this is the same case as before, but it is indicated
                            # by a nan value at the second index. Therefore
                            # there are no properties defined on which the
                            # threshold depends (but other properties have more
                            #  values defined, which is why the dimension in the
                            #  array is longer, therefore min_value is longer)
                            threshold = threshold
                        else:
                            min_value_nb = 3
                            while min_value_nb < len(min_value):
                                val_property_nb = min_value[min_value_nb]
                                if math.isnan(val_property_nb):
                                    break
                                # if property values are added, the threshold should
                                # be reduced by the other property values
                                # (since higher other property values means that the
                                # threshold "is reached earlier")
                                if min_value[2] == 1:
                                    threshold = threshold - properties_array[
                                        0, int(val_property_nb), object_pos,
                                        sim_id, param_id]
                                if min_value[2] == -1:
                                    threshold = threshold + properties_array[0, int(val_property_nb),
                                                                             object_pos,
                                                                             sim_id,
                                                                             param_id]
                                min_value_nb += 1

                        # if threshold was defined as None, it is now NaN
                        # and therefore the comparison will always be False
                        if new_property_val < threshold:
                            new_property_val = threshold

                            # new_property_val = math.nan
                            # # HARD CODE LOSS OF MT UPON REACHING 0 length
                            # object_states[0, object_pos,
                            #                  sim_id, param_id] = 0
                            # properties_array[0, 0,
                            #                  object_pos,
                            #                  sim_id,
                            #                  param_id] = math.nan
                            #
                            # properties_array[0, 1,
                            #                  object_pos,
                            #                  sim_id,
                            #                  param_id] = math.nan
                            #
                            # properties_array[0, 2,
                            #                  object_pos,
                            #                  sim_id,
                            #                  param_id] = math.nan
                            #
                            # cuda.atomic.add(nb_objects_all_states,
                            #                 (0, 2,
                            #                       sim_id, param_id), -1)
                            #
                            # cuda.atomic.add(nb_objects_all_states,
                            #                 (0, 0,
                            #                       sim_id, param_id), 1)

                    # set the property val to the property val within the
                    # [min_value, max_value] limits
                    # if math.isnan(new_property_val) & (core_id == 0) & (sim_id == 0):
                    #     print(900, threshold, max_value[0], max_value[1])

                    # if (core_id == 0) & (sim_id == 0):
                    #     print(properties_array[0, int(property_nb),
                    #                  object_pos,
                    #                  sim_id, param_id], new_property_val)

                    # if (int(property_nb) > 0) & (new_property_val > 30):
                    #     print(111, object_states[0, object_pos, sim_id, param_id],
                    #           new_property_val,
                    #           properties_array[0, int(property_nb), object_pos,
                    #                            sim_id, param_id],
                    #           threshold)

                    # if (sim_id == 0) & (param_id == 0) & (core_id == 0) & (times[0, sim_id, param_id] < 0.2):
                    #     print(55, property_nb, new_property_val, property_val,
                    #           threshold, max_value[0],
                    #           len(max_value), max_value[1])

                    # if object_states[0, object_pos,
                    #                      sim_id, param_id] > 0:
                    _track_object_property_changes(object_pos, property_nb,
                                                   new_property_val,
                                                   properties_array,
                                                   local_object_lifetime_array,
                                                 local_lifetime_resolution,
                                                   times,
                                                   sim_id, param_id)
                    properties_array[0, int(property_nb),
                                     object_pos,
                                     sim_id, param_id] = new_property_val

                    action_property_nb += 1
                object_pos += 1
        action_nb += 1

    return

def _get_random_mt_at_position(position, object_states, properties_array,
                                first_last_idx_with_object, core_id, sim_id,
                               param_id,
                                rng_states, thread_masks, nb_parallel_cores,
                                simulation_factor, parameter_factor):

    # have array with same size as object_state array
    # to save the object_numbers of all target objects
    # the array has the size (object_nb + 33, sims, params)
    target_object_nbs = 0

    nb_objects = first_last_idx_with_object[1,sim_id, param_id] + 1

    nb_objects_per_core = int(nb_objects / nb_parallel_cores[sim_id, param_id])
    (object_pos,
     last_object_pos) = _get_first_and_last_object_pos(nb_objects,
                                                       nb_parallel_cores[
                                                           sim_id, param_id],
                                                       core_id)
    nb_of_target_objects = 0
    target_state = 3
    while object_pos < last_object_pos:
        if object_states[0, object_pos, sim_id, param_id] == target_state:
            # check whether the starting point of the object is before the
            # target position
            object_pos = properties_array[0, object_pos, sim_id, param_id]
            if object_pos < position:
                property_nb = 1
                while property_nb < properties_array.shape[0]:
                    object_pos += properties_array[property_nb, object_pos,
                                                 sim_id, param_id]
                if object_pos >= position:
                    # if the end position of the object is at least at
                    # the target position, then the current object is one of
                    # the target objects
                    target_object_nbs[nb_objects_per_core * core_id +
                                      nb_of_target_objects,
                                      sim_id, param_id] = object_pos
                    nb_of_target_objects += 1
                    # add one to the total count of objects
                    cuda.atomic.add(target_object_nbs,
                                    (-1, sim_id, param_id), 1)
                    # add one to the count of objects for the current thread
                    target_object_nbs[-core_id - 2,
                                      sim_id, param_id] += 1

    if core_id == 0:
        # get the idx of the target object among all potential target objects
        random_object_idx = int(math.floor(_get_random_number(sim_id, param_id,
                                                           rng_states,
                                                           simulation_factor,
                                                           parameter_factor) *
                                       target_object_nbs[-1, sim_id, param_id]))
        # check which of the threads (core_ids) found the target object
        # by starting from the first thread
        # (core_id = 0, idx in target_object_nbs = -2)
        # and going through all the following ones and checking when the sum of
        # potential target objects goes above the random_object_nb
        process_id = -2
        object_nb_sum = 0
        while abs(process_id)-1 <= nb_parallel_cores[sim_id, param_id]:
            object_nb_sum += target_object_nbs[process_id,
                                               sim_id, param_id]
            if (object_nb_sum - 1) >= random_object_idx:
                break
            process_id -= 1

        # the actual object_nb then is at the index
        # core_id * nb_objects_per_core (start of all object_nbs of potential
        # targets)
        # + object_nb_sum - random_object_nb -
        # target_object_nbs[process_id, sim_id, param_id]
        # (the final random object idx starting to count from the beginning of
        # the range of the target thread)
        target_object_nbs[-1, sim_id,
                          param_id] = target_object_nbs[(abs(process_id) - 3) *
                                                        nb_objects_per_core +
                                                        (target_object_nbs[process_id + 1,
                                                                           sim_id,
                                                                           param_id]
                                                         - (object_nb_sum -
                                                         random_object_idx)
                                                         ),
                                                        sim_id, param_id]
    cuda.syncwarp(thread_masks[sim_id, param_id])
    return

def _get_density_dependent_position(transition_nb, creation_on_objects,
                                    local_density, total_density,
                                    local_resolution, rng_states,
                                    simulation_factor, parameter_factor,
                                    sim_id, param_id, core_id):
        # check if only one density was measured (last index only NaN,
        # which means that all density dependent transitions rely on the total
        # density (of all properties) and not on densities of a subset of
        # properties
        all_density = 0
        if math.isnan(creation_on_objects[-1, 0, 0]):
            all_density += total_density[0, sim_id, param_id]
        else:
            # if more than one density was measured, add the densities that
            # the current transition dependes on
            property_nb = 0
            while property_nb < creation_on_objects.shape[2]:
                if math.isnan(creation_on_objects[transition_nb, 0,
                                                  property_nb]):
                    break
                all_density += total_density[int(creation_on_objects[
                                                   transition_nb, 0,
                                                   property_nb]),
                                           sim_id, param_id]
                property_nb += 1

        # get threshold by multiplying total density by random number
        # from 0 to 1
        threshold = all_density * _get_random_number(sim_id, param_id,
                                       rng_states,
                                       simulation_factor,
                                       parameter_factor)

        # go through local density until threshold is reached
        x_pos = 0
        density_sum = 0
        while x_pos < local_density.shape[1]:
            local_density_here = 0
            # sum densities relevant to current transition
            if math.isnan(creation_on_objects[-1, 0, 0]):
                local_density_here += local_density[0, x_pos,
                                                   sim_id, param_id]
            else:
                property_nb = 0
                while property_nb < creation_on_objects.shape[2]:
                    if math.isnan(creation_on_objects[transition_nb, 0,
                                                      property_nb]):
                        break
                    local_density_here += local_density[int(
                        creation_on_objects[transition_nb,
                                            0, property_nb]),
                                                        x_pos, sim_id, param_id]

                    if local_density[int(
                        creation_on_objects[transition_nb,
                                            0, property_nb]),
                                                        x_pos, sim_id, param_id] < 0:
                        print(345)
                    property_nb += 1

            if local_density_here < 0:
                print(2345)

            density_sum = density_sum + local_density_here
            if round(density_sum,6) >= round(threshold,6):
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
                property_val -= (((density_sum - threshold) /
                                 local_density_here) *
                                 local_resolution)
                return x_pos, property_val, local_density_here
            x_pos += 1

        if core_id == 0:
            print(123456, local_density.shape[1], x_pos, density_sum, threshold,
                  all_density, sim_id, param_id)

        return x_pos, math.nan, local_density_here


def _update_object_states(current_transitions, all_transition_states,
                          all_transition_positions, object_states,
                          nb_objects_all_states,
                          properties_array ,
                          transition_parameters,
                          all_transition_tranferred_vals,
                          all_transition_set_to_zero_properties,
                          property_start_values,
                            changed_start_values_array,
                            creation_on_objects,
                          first_last_idx_with_object,

                          local_object_lifetime_array,
                          local_lifetime_resolution,

                          parameter_value_array, params_prop_dependence,
                          property_extreme_values, timepoint_array,
                          current_transition_rates,

                          local_density, total_density,
                          local_resolution,
                          density_threshold_boundaries,
                          nb_parallel_cores, thread_masks, core_id,
                          sim_id, param_id,
                          rng_states, simulation_factor, parameter_factor,

                          times):

    # update the simulations according to executed transitions
    transition_nb = current_transitions[sim_id, param_id]
    if not math.isnan(transition_nb):
        transition_nb = int(transition_nb)
        transition_states = all_transition_states[transition_nb]
        start_state = transition_states[0]
        end_state = transition_states[1]
        transition_position = int(all_transition_positions[sim_id, param_id])
        if math.isnan(properties_array[0, 0, transition_position,
                                       sim_id, param_id]) & (start_state != 0):
            print(111)
            return
        # end_pos = properties_array[0, 0, transition_position, sim_id, param_id] + properties_array[0, 1, transition_position, sim_id, param_id]
        # if end_pos == 20:
        #     return

        object_states[0, transition_position,sim_id, param_id] = end_state
        # change the object counter according to the executed transition
        # don't apply standard changes for cutting
        # (creation on object[n, 1,0] not nan))
        # since changes in object numbers are more complicated due to
        # different object states being differently affected by cutting
        if (core_id == 0) & (math.isnan(creation_on_objects[transition_nb,
                                                             1, 0])):
            if start_state != 0:
                nb_objects_all_states[0, int(start_state),
                                      sim_id, param_id] -= 1
            else:
                # nb_objects_all_states[0, 0, sim_id, param_id] -= 1

                # if resources for the object generation transition are defined,
                # set generation method for object
                if not math.isnan(transition_parameters[int(transition_nb), 1]):
                    # set as parameter number of resource for transition + 1 so
                    # that 0 indicates not generated by a resource limited
                    # object generation
                    object_states[1, transition_position,
                                  sim_id, param_id] = transition_nb + 1
                    nb_objects_all_states[1, int(transition_parameters[
                                                     transition_nb, 1]),
                                          sim_id, param_id] += 1
                # if the current transition should be tracked (and inherited to
                # MTs forming on top of it),
                # set the transition number for the current MT
                if transition_parameters[int(transition_nb), 2] == 1:
                    object_states[2, transition_position,
                                  sim_id, param_id] = transition_nb + 1

            if end_state != 0:
                nb_objects_all_states[0, int(end_state), sim_id, param_id] += 1
            else:
                # nb_objects_all_states[0, 0, sim_id, param_id] += 1
                # set generation method of object to 0 since the object is
                # removed
                # to get the generation method, use the object state in the
                # second dimension
                if object_states[1, transition_position,
                                 sim_id, param_id] != 0:
                    # object states save the transition number,
                    # not the parameter number directly
                    # therefore the parameter number has to be obtained from
                    # transition_parameters.
                    if not math.isnan(
                            transition_parameters[int(object_states[
                                                          1, transition_position,
                                                          sim_id, param_id] - 1),
                                                  1]):
                        nb_objects_all_states[1,
                                              int(transition_parameters[
                                                      object_states[
                                                          1, transition_position,
                                                          sim_id, param_id] - 1, 1]),
                                              sim_id, param_id] -= 1
                    object_states[1, transition_position,
                                  sim_id, param_id] = 0

                if object_states[2, transition_position, sim_id, param_id] != 0:
                    object_states[2, transition_position, sim_id, param_id] = 0

        # if objects are cut creation_on_objects for the transition is not
        # nan at idx 1 and 2 and contain transition maps at these indices
        # indicating that the transition includes cutting of objects
        if not math.isnan(creation_on_objects[transition_nb, 1, 0]):
            if core_id == 0:
                _cut_object(transition_position, transition_nb,
                            transition_parameters,
                            creation_on_objects,
                            object_states, nb_objects_all_states,
                            properties_array, first_last_idx_with_object,
                            local_density, total_density, local_resolution,
                            local_lifetime_resolution, local_object_lifetime_array, times,
                            rng_states, simulation_factor, parameter_factor,
                            sim_id, param_id, core_id)

        # change property values based on transitions
        elif start_state == 0:

            params_prop_dependence = params_prop_dependence[
                int(transition_parameters[transition_nb, 0])]
            # check if the idx for creating a new object is higher than
            # the currently highest idx
            if transition_position > first_last_idx_with_object[1, sim_id,
                                                                param_id]:

                first_last_idx_with_object[1, sim_id,
                                           param_id] = transition_position

            # if a new object was created, set property values according to
            # defined value
            nb_properties = property_start_values.shape[0]
            (property_nb,
             last_property_nb) = _get_first_and_last_object_pos(
                nb_properties, nb_parallel_cores[sim_id, param_id], core_id)

            # set start of lifetime for objects
            local_object_lifetime_array[transition_position, 0,
                                        sim_id, param_id] = 0

            # property_nb = 0
            # while property_nb < property_start_values.shape[0]:
            while property_nb < last_property_nb:

                if ((not math.isnan(creation_on_objects[
                                        transition_nb, 0, 0])) &
                        (property_nb == 0)):

                    # One more arrays needed:
                    # density_threshold_boundaries (3, sim_id, param_id)
                    (x_pos, property_val,
                     local_density_here) = _get_density_dependent_position(
                        transition_nb, creation_on_objects,
                        local_density, total_density, local_resolution,
                        rng_states, simulation_factor, parameter_factor,
                        sim_id, param_id, core_id)

                    # if the current transition allows inheritance of
                    # orientation, inherit orientation
                    # (index 2 of object_states)
                    if transition_parameters[transition_nb, 2] == 2:
                        target_property_nbs = creation_on_objects[transition_nb,
                                                                  0]

                        (template_object_position,
                         _, _) = _get_random_object_at_position(
                            target_property_nbs, x_pos,
                            local_density_here, object_states, properties_array,
                            first_last_idx_with_object, local_resolution,
                            rng_states, simulation_factor, parameter_factor,
                            sim_id, param_id, core_id)

                        if object_states[2, template_object_position,
                                         sim_id, param_id] != 0:
                            object_states[2, transition_position,
                                          sim_id, param_id] = object_states[
                                2, template_object_position, sim_id, param_id]

                    # # check whether property value was calculated correctly
                    # if core_id == 0:
                    #     if round(property_val,4) != round(property_val_new,4):
                    #         print(999999, sim_id, param_id,
                    #               x_pos-1, final_position,
                    #               property_val, property_val_new)

                else:
                    # check whether the property start values are different for the
                    # current transition
                    if math.isnan(changed_start_values_array[transition_nb,
                                                             property_nb, 0]):
                        property_start_val = property_start_values[property_nb]
                    else:
                        property_start_val = changed_start_values_array[
                            transition_nb,
                            property_nb]
                    # if there is only one non-nan value or just one value in total
                    # then the first value is the actual value that the property
                    # should be set to
                    if len(property_start_val) == 1:
                        property_val = property_start_val[0]
                    elif math.isnan(property_start_val[1]):
                        property_val = property_start_val[0]
                    elif not math.isnan(params_prop_dependence[2]):
                        # POSITION DEPENDENT NUCLEATION IS ONLY IMPLEMENTED
                        # FOR LINEAR POSITION DEPENDENCE SO FAR!!!
                        random_nb = _get_random_number(sim_id, param_id,
                                                       rng_states,
                                                       simulation_factor,
                                                       parameter_factor)

                        # if the creation of objects is property dependent
                        # (only position dependence is possible since it does
                        #  not depend on other objects)
                        # calculate the position based on this dependence

                        (total_dependence_rate, max_property_change,
                         max_position,
                         start_value,
                         base_value,
                         change_per_um) = _get_total_pos_dependent_creation_rate(
                                transition_nb,
                                params_prop_dependence,
                               parameter_value_array,
                               property_extreme_values,
                               transition_parameters,
                               timepoint_array, sim_id, param_id,
                               )

                        threshold = random_nb * current_transition_rates[
                            transition_nb, sim_id, param_id]

                        # if dependence is from the end, the threshold starts
                        # counting from the end as well
                        # if dependence is from the start, the threshold starts
                        # founding from the start as well
                        # therefore whenever the threshold is smaller than the
                        # total rate, it is within the dependence range

                        if threshold < total_dependence_rate:
                            # if it is within the property dependence, solve
                            # the quadratic equation to get the point corresponding
                            # to the random threshold
                            property_val = - ((start_value + base_value) /
                                              change_per_um)
                            property_val_1 = (property_val -
                                              math.sqrt(((start_value +
                                                          base_value) /
                                                         change_per_um)**2
                                                        + 2 *
                                                        threshold /
                                                        change_per_um))
                            property_val_2 = (property_val +
                                              math.sqrt(((start_value +
                                                          base_value) /
                                                         change_per_um)**2
                                                        + 2 *
                                                        threshold /
                                                        change_per_um))
                            if property_val_1 > 0:
                                property_val = property_val_1
                            else:
                                property_val = property_val_2
                        else:
                            # if it is outside of the property dependence, solve
                            # the linear equation to get the point corresponding
                            # to the random threshold
                            property_val = ((threshold - total_dependence_rate)
                                            / base_value) + max_property_change

                        # # since for dependence from end, threshold starts from
                        # # the end, subtract the calculated position from the
                        # # maximum position to get the actual position
                        if math.isnan(params_prop_dependence[0]):
                            property_val = max_position - property_val

                        # print(property_val, threshold, max_property_change,
                        #       total_dependence_rate,
                        #       property_val_1, property_val_2,
                        #       current_transition_rates[
                        #           transition_nb, sim_id, param_id],
                        #       base_value, start_value, max_position)

                    else:
                        # if there are two non-nan start property vals, then
                        # the property val should be a random number between these
                        # two numbers, with the first number indicating the start
                        # (lower value) and the second number the end of the
                        # interval (higher value)
                        random_nb = _get_random_number(sim_id, param_id,
                                                       rng_states,
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


                # if (int(property_nb) > 0) & (property_val > 30):
                #     print(222, object_states[0, int(transition_position), sim_id, param_id],
                #           property_val,
                #           properties_array[0, int(property_nb), int(transition_position),
                #                            sim_id, param_id])

                # if (sim_id == 0) & (param_id == 0) & (core_id == 0) & (times[0, sim_id, param_id]< 0.4):
                #     print(33, transition_position, property_nb, property_val)

                _track_object_property_changes(transition_position,
                                               property_nb,
                                               property_val,
                                               properties_array,
                                               local_object_lifetime_array,
                                               local_lifetime_resolution,
                                               times,
                                               sim_id, param_id)

                properties_array[0, property_nb,
                                 int(transition_position),
                                 sim_id, param_id] = property_val

                property_nb += 1

        elif (end_state == 0) & (core_id == 0):
            # if the removed position is smaller than the first idx with object
            # set the removed position as smallest number (as approximation)

            if core_id == 0:
                if transition_position < first_last_idx_with_object[0,sim_id,
                                                                    param_id]:
                    first_last_idx_with_object[0,sim_id,
                                               param_id] = transition_position

            # if an object was removed, set property values to NaN
            # nb_properties = property_start_values.shape[0]
            # (property_nb,
            #  last_property_nb) = _get_first_and_last_object_pos(nb_properties,
            #                                                   nb_parallel_cores[sim_id, param_id],
            #                                                   core_id)
            property_nb = property_start_values.shape[0] - 1
            while property_nb >= 0:
                _track_object_property_changes(transition_position,
                                               property_nb,
                                               0,
                                               properties_array,
                                               local_object_lifetime_array,
                                                 local_lifetime_resolution,
                                                   times,
                                               sim_id, param_id)
                properties_array[0, property_nb,
                                 int(transition_position),
                                 sim_id, param_id] = math.nan
                property_nb -= 1

        # if the object is not removed, it might be that the transition triggers
        # other events as well
        if (end_state != 0) & (core_id == 0):
            # if property values are transfered, then the first number
            # is the property number source and the second is the target
            # if the value is nan then there is no transfer
            transfered_vals = all_transition_tranferred_vals[int(transition_nb)]

            if math.isnan(transfered_vals[0]) == False:
                source_property_number = transfered_vals[0]
                target_property_number = transfered_vals[1]
                source_val = properties_array[0, int(source_property_number),
                                              int(transition_position),
                                              sim_id, param_id]
                target_val = properties_array[0, int(target_property_number),
                                              int(transition_position),
                                              sim_id, param_id]

                # _track_object_property_changes(int(transition_position),
                #                                int(source_property_number),
                #                                0,
                #                                properties_array,
                #                                local_object_lifetime_array,
                #                                  local_lifetime_resolution,
                #                                    times,
                #                                sim_id, param_id)
                properties_array[0, int(source_property_number),
                                 int(transition_position),
                                 sim_id, param_id] = 0 #source_val/2

                # _track_object_property_changes(int(transition_position),
                #                                int(target_property_number),
                #                                target_val + source_val,
                #                                properties_array,
                #                                local_object_lifetime_array,
                #                                  local_lifetime_resolution,
                #                                    times,
                #                                sim_id, param_id)
                properties_array[0, int(target_property_number),
                                 int(transition_position),
                                 sim_id, param_id] = target_val + source_val #/2


            # if properties should be set to zero for the current transition
            # do that at the current position
            set_to_zero_properties = all_transition_set_to_zero_properties[
                int(transition_nb)]
            if math.isnan(set_to_zero_properties[0]) == False:
                zero_property_nb = 0
                while zero_property_nb < len(set_to_zero_properties):
                    zero_property = set_to_zero_properties[zero_property_nb]
                    if math.isnan(zero_property):
                        break
                    _track_object_property_changes(int(transition_position),
                                                   int(zero_property),
                                                   0,
                                                   properties_array,
                                                   local_object_lifetime_array,
                                                 local_lifetime_resolution,
                                                   times,
                                                   sim_id, param_id)
                    properties_array[0, int(zero_property),
                                     int(transition_position),
                                     sim_id, param_id] = 0
                    zero_property_nb += 1

    return None


def _get_total_pos_dependent_creation_rate(transition_nb,
                                           params_prop_dependence,
                                           parameter_value_array,
                                           property_extreme_values,
                                           transition_parameters,
                                           timepoint_array, sim_id, param_id,
                                           ):

    max_position = parameter_value_array[
        int(property_extreme_values[1, 0, 0]),
        int(timepoint_array[0, sim_id, param_id]),
        param_id]

    # Property dependence might only be over a limited
    # range of positions. Therefore, check if the random
    # threshold is within the property dependence.
    if math.isnan(params_prop_dependence[0]):
        start_value = parameter_value_array[int(params_prop_dependence[1]),
                                           int(timepoint_array[0, sim_id,
                                                               param_id]),
                                            param_id]
    else:
        start_value = parameter_value_array[int(params_prop_dependence[0]),
                                            int(timepoint_array[0, sim_id,
                                                                param_id]),
                                           param_id]

    if math.isnan(params_prop_dependence[4]):
        change_per_um = (parameter_value_array[int(params_prop_dependence[1]),
                                               int(timepoint_array[0, sim_id,
                                                                   param_id]),
                                               param_id] -
                     parameter_value_array[int(params_prop_dependence[0]),
                                           int(timepoint_array[0, sim_id,
                                                               param_id]),
                                           param_id])
        if params_prop_dependence[3] == 0:
            change_per_um /= max_position
    else:
        change_per_um = parameter_value_array[int(params_prop_dependence[4]),
                                              int(timepoint_array[0, sim_id,
                                                                  param_id]),
                                              param_id]


    base_value = parameter_value_array[
                     int(transition_parameters[transition_nb, 0]),
                     int(timepoint_array[0, sim_id,param_id]),
                     param_id] / max_position

    if (start_value / change_per_um) < 0:
        max_property_change =  - (start_value /
                                  change_per_um)
    else:
        max_property_change = max_position


    # calculate the total transition rate within the
    # depedence
    total_dependence_rate = (max_property_change *
                             (start_value + base_value))
    total_dependence_rate += (change_per_um/2 *
                              max_property_change ** 2)

    return (total_dependence_rate, max_property_change,
            max_position, start_value, base_value, change_per_um)


def _cut_object(transition_position, transition_nb,
                transition_parameters,
                creation_on_objects,
                object_states, nb_objects_all_states,
                properties_array, first_last_idx_with_object,
                local_density, total_density, local_resolution,
                local_lifetime_resolution, local_object_lifetime_array, times,
                rng_states, simulation_factor, parameter_factor,
                sim_id, param_id, core_id):
    # FUR CUTTING LIFETIME OF OBJECTS SHOULD RATHER BE TRANSFERRED
    # BUT RIGHT NOW NEW TIMESTAMPS ARE USED INSTEAD; WHICH IS WRONG!
    # get the x position of the cut
    x_pos, _, local_density_here = _get_density_dependent_position(
        transition_nb, creation_on_objects,
        local_density, total_density, local_resolution,
        rng_states, simulation_factor, parameter_factor,
        sim_id, param_id, core_id)

    target_property_nbs = creation_on_objects[transition_nb, 0]

    (object_position,
     length_in_range,
     start) = _get_random_object_at_position(
        target_property_nbs, x_pos,
        local_density_here, object_states, properties_array,
        first_last_idx_with_object, local_resolution,
        rng_states, simulation_factor, parameter_factor,
        sim_id, param_id, core_id)


    # elif (sim_id == 2) & (core_id == 0):
    #     print(0)

    # now determine random position within the range
    # at which the object is cut
    # the length_in_range can be max of 1, if the entire range
    # of the x position is covered
    cut_length = (_get_random_number(sim_id, param_id,
                                    rng_states,
                                    simulation_factor,
                                    parameter_factor) *
                  length_in_range)

    # add the previous property value to that
    # length in range, but don't allow it to be reduced
    # (e.g. if start is after the beginning of the range)
    cut_length += max(0, ((x_pos-1) * local_resolution - start))

    # cutting has to be done on one specific property
    # reduce this property by cut_length
    property_nb = int(creation_on_objects[transition_nb, 0, 0])

    # make sure cut length is not larger than the property value
    if properties_array[0, property_nb,
                   object_position, sim_id, param_id] < cut_length:
        print(55555, properties_array[0, property_nb,
                   object_position, sim_id, param_id], cut_length,
              start, x_pos,
              (x_pos-1) * local_resolution)

    # initialize all properties of new object as 0 (was nan before)
    property_nb_tmp = 0
    while property_nb_tmp < properties_array.shape[1]:

        _track_object_property_changes(transition_position,
                                       property_nb_tmp,
                                       0,
                                       properties_array,
                                       local_object_lifetime_array,
                                     local_lifetime_resolution,
                                       times,
                                       sim_id, param_id)
        properties_array[0, property_nb_tmp,
                       transition_position,
                       sim_id,
                       param_id] = 0
        property_nb_tmp += 1

    # property_nb_tmp = 0
    # while property_nb_tmp < properties_array.shape[1]:
    #     print(1, property_nb_tmp,
    #           properties_array[0, property_nb_tmp, transition_position,
    #                      sim_id, param_id])
    #     print(2, property_nb_tmp,
    #           properties_array[0, property_nb_tmp, object_position,
    #                      sim_id, param_id])
    #     property_nb_tmp += 1

    _track_object_property_changes(object_position,
                                   property_nb,
                                   properties_array[
                                       0, property_nb,
                                       object_position,
                                       sim_id,
                                       param_id] - cut_length,
                                   properties_array,
                                   local_object_lifetime_array,
                                     local_lifetime_resolution,
                                       times,
                                   sim_id, param_id)
    properties_array[0, property_nb,
                     object_position,
                     sim_id, param_id] -= cut_length

    if cut_length < 0:
        print(400000, cut_length)

    # the new object will get the cut length as new property value

    _track_object_property_changes(transition_position,
                                   property_nb,
                                   cut_length,
                                   properties_array,
                                   local_object_lifetime_array,
                                     local_lifetime_resolution,
                                       times,
                                   sim_id, param_id)
    properties_array[0, property_nb,
                     transition_position,
                     sim_id, param_id] = cut_length

    # print(111, object_position, property_nb, cut_length, start, x_pos, local_resolution, length_in_range)
    # transfer values of all properties before the
    # current property (since the order of properties defines the
    # physical order of properties in the object) to the new object
    # and set those properties to zero in the original
    # object (do it differently for the position though,
    # property_nb == 0)
    property_nb -= 1
    while property_nb > 0:
        # set the property value of the new object
        # to the value of the old object

        # _track_object_property_changes(transition_position,
        #                                property_nb,
        #                                properties_array[
        #                                    0, property_nb,
        #                                    object_position,
        #                                    sim_id, param_id],
        #                                properties_array,
        #                                local_object_lifetime_array,
        #                              local_lifetime_resolution,
        #                                times,
        #                                sim_id, param_id)
        properties_array[0, property_nb,
                       transition_position,
                       sim_id,
                       param_id] = properties_array[0, property_nb,
                                                  object_position,
                                                  sim_id, param_id]

        _track_object_property_changes(object_position,
                                       property_nb,
                                       0,
                                       properties_array,
                                       local_object_lifetime_array,
                                     local_lifetime_resolution,
                                       times,
                                       sim_id, param_id)
        # set the property value of the old object to 0
        properties_array[0, property_nb,
                       object_position,
                       sim_id, param_id] = 0
        property_nb -= 1

    # the new object will have the position of the old object
    properties_array[0, 0,
                   transition_position,
                   sim_id,
                   param_id] = properties_array[0, 0,
                                              object_position,
                                              sim_id, param_id]

    # position of the old object will be set to the cut position
    properties_array[0, 0, object_position,
                   sim_id, param_id] = cut_length + start

    # # check that no changed object is beyond hard coded maximum
    # # of 20 (for max dimension of 20)
    # property_nb_tmp = 0
    # length_tmp = 0
    # while property_nb_tmp < properties_array.shape[1]:
    #     length_tmp += properties_array[0, property_nb_tmp,
    #                                    transition_position,
    #                                    sim_id, param_id]
    #     property_nb_tmp += 1
    #
    # if length_tmp > 20:
    #     print(200000, length_tmp)
    #
    # property_nb_tmp = 0
    # length_tmp = 0
    # while property_nb_tmp < properties_array.shape[1]:
    #     length_tmp += properties_array[0, property_nb_tmp,
    #                                    object_position,
    #                                    sim_id, param_id]
    #     property_nb_tmp += 1
    #
    # if length_tmp > 20:
    #     print(300000, length_tmp,
    #           properties_array[0, 0,
    #                            object_position,
    #                            sim_id, param_id],
    #           properties_array[0, 1,
    #                            object_position,
    #                            sim_id, param_id],
    #           properties_array[0, 2,
    #                            object_position,
    #                            sim_id, param_id],
    #           object_position, start,
    #           cut_length,# tmp_prop, tmp_prop2,
    #           x_pos
    #           )

    # property_nb_tmp = 0
    # while property_nb_tmp < properties_array.shape[1]:
    #     print(11, property_nb_tmp,
    #           properties_array[0, property_nb_tmp, transition_position,
    #                      sim_id, param_id])
    #     print(22, property_nb_tmp,
    #           properties_array[0, property_nb_tmp, object_position,
    #                      sim_id, param_id])
    #     property_nb_tmp += 1

    # reduce object number for state of the cut object
    nb_objects_all_states[0, object_states[0, int(object_position),
                                           sim_id, param_id],
                          sim_id, param_id] -= 1

    # increase object number for target states of object
    # Before cut is the new object (new tip position)
    state_before_cut = creation_on_objects[transition_nb, 1,
                                           object_states[
                                               0, int(object_position),
                                               sim_id, param_id]-1]
    # After cut is the old object (same tip position)
    state_after_cut = creation_on_objects[transition_nb, 2,
                                           object_states[
                                               0, int(object_position),
                                               sim_id, param_id]-1]

    if state_before_cut == 0:
        print(777770)
        property_nb_tmp = properties_array.shape[1]-1
        while property_nb_tmp >= 0:
            _track_object_property_changes(transition_position,
                                           property_nb_tmp,
                                           0,
                                           properties_array,
                                           local_object_lifetime_array,
                                           local_lifetime_resolution,
                                           times, sim_id, param_id)
            properties_array[0, property_nb_tmp,
                           transition_position,
                           sim_id,
                           param_id] = math.nan
            property_nb_tmp -= 1
        # object_states[0, transition_position, sim_id, param_id] = 0
        # if transition_position == first_last_idx_with_object[1, ]

    if state_after_cut == 0:
        print(777771)
        property_nb_tmp = properties_array.shape[1] - 1
        while property_nb_tmp > 0:

            _track_object_property_changes(object_position,
                                           property_nb_tmp,
                                           0,
                                           properties_array,
                                           local_object_lifetime_array,
                                           local_lifetime_resolution,
                                           times, sim_id, param_id)
            properties_array[0, property_nb_tmp,
                           object_position,
                           sim_id,
                           param_id] = math.nan
            property_nb_tmp -= 1
        nb_objects_all_states[0,
                              int(object_states[0, object_position,
                                                sim_id, param_id]),
                              sim_id, param_id] -= 1
        object_states[0, object_position, sim_id, param_id] = 0
    else:
        nb_objects_all_states[0, int(state_after_cut),
                              sim_id, param_id] += 1


        # only if resources are defined for the current transition,
        # increment the counter
        if not math.isnan(transition_parameters[transition_nb, 1]):
            nb_objects_all_states[1, int(transition_parameters[transition_nb, 1]),
                                  sim_id, param_id] += 1

            object_states[1, transition_position,
                          sim_id, param_id] = (transition_nb + 1)

        # # if the current transition should be tracked (and inherited to
        # # MTs forming on top of it),
        # # set the transition number for the current MT
        # if transition_parameters[int(transition_nb), 2] == 1:
        #     object_states[2, transition_position,
        #                   sim_id, param_id] = transition_nb + 1

    # print(333, state_before_cut, state_after_cut,
    #       transition_position, object_states[0, transition_position,
    #               sim_id, param_id],
    #       object_position,object_states[0, object_position,
    #               sim_id, param_id],  density_sum, threshold)
    # change object state of old object (that was cut)

    object_states[0, object_position,
                  sim_id, param_id] = state_after_cut


    # change object state of new object
    # (new object end(tip) that was formed)
    object_states[0, transition_position,
                  sim_id, param_id] = state_before_cut

    nb_objects_all_states[0, int(state_before_cut),
                          sim_id, param_id] += 1

    # if length_tmp > 20:
    #     print(300001,
    #           object_states[0, object_position, sim_id, param_id],
    #           object_states[0, transition_position,sim_id, param_id]
    #           )
    # check if the idx for creating a new object is higher than
    # the currently highest idx
    if transition_position > first_last_idx_with_object[1, sim_id, param_id]:
        # print(5551, transition_position)
        first_last_idx_with_object[1,sim_id,
                                   param_id] = transition_position

    # if (first_last_idx_with_object[1,sim_id,param_id] == 0):
    #     print(66666, first_last_idx_with_object[1,sim_id,param_id])


def _get_random_object_at_position(target_property_nbs, x_pos,
                                   local_density_here,
                                   object_states, properties_array,
                                   first_last_idx_with_object,
                                   local_resolution,
                                   rng_states,
                                   simulation_factor, parameter_factor,
                                   sim_id, param_id, core_id):
    # get the object that was cut at that position
    # by first finding a random threshold, depending on the local
    # density at that position
    threshold = (_get_random_number(sim_id, param_id,
                                   rng_states,
                                   simulation_factor,
                                   parameter_factor) *
                 local_density_here)

    density_sum = 0
    # Go through each object, check whether it is in range
    # and then check whether adding its density in the range
    # pushes the density sum above the threshold
    object_position = 0
    while object_position <= first_last_idx_with_object[1, sim_id, param_id]+20:
        if object_states[0, object_position, sim_id, param_id] > 0:
            # check if property value is zero

            # if properties_array[0, target_property_nb,
            #                   object_position,
            #                   sim_id, param_id] > 0:
            property_nb = 0
            # start position is the sum of all properties before
            # the target property, for the end position
            # add the target property to that
            start = 0
            while property_nb < target_property_nbs[0]:
                start += properties_array[0, property_nb, object_position,
                                          sim_id, param_id]
                property_nb += 1

            # start has to be before the end of the x pos range
            if start < ((x_pos + 1) * local_resolution):

                end = start
                # index = 1
                # while index < properties_array.shape[1]:
                #     property_nb = index
                index = 0
                while index < target_property_nbs.shape[0]:
                    property_nb = target_property_nbs[index]
                    if math.isnan(property_nb):
                        break
                    # its important to only convert to int now, otherwise the
                    # int conversion of a nan is unpredictable and can lead to
                    # weird results at seemingly random objects
                    property_nb = int(property_nb)
                    if not math.isnan(properties_array[0, property_nb,
                                                       object_position,
                                                       sim_id, param_id]):
                        end += properties_array[0, property_nb, object_position,
                                                sim_id, param_id]
                    index += 1

                # end has to be after the beginning of the range
                if end > ((x_pos-1) * local_resolution):
                    # end = start + properties_array[0, target_property_nb,
                    #                                object_position,
                    #                                sim_id, param_id]
                    start_min = max(-local_resolution, start)
                    if (end - start_min) > 0:
                        # if ((end > (x_pos - 1) * local_resolution)):
                        # increase density_sum by fraction of
                        # the range filled by the property values
                        # of the current object
                        # allow a maximum of 1 to be added
                        # if end is until the end of the range

                        density = 1
                        # the start point is not actually at the very
                        # beginning of the resolution but after that
                        # therefore the added density is below 1 for the
                        # first position. Subtract the relative amount of
                        # the local resolution that the object starts
                        # after the x_pos

                        density -= max(0,min(1,((start -
                                     x_pos * local_resolution) /
                                    local_resolution)))
                        # density -= ( 1 -
                        #              (x_pos * local_resolution - start)
                        #              / local_resolution)
                        # for the x bin in which the MT ended, don't add a
                        # full MT but just the relative amount of the bin
                        # crossed by the MT
                        # x_um = x_pos * local_resolution
                        # density -= ((x_pos * local_resolution - end) / local_resolution)
                        density -= max(0, min(1,(1 - ((end - x_pos *
                                          local_resolution) /
                                         local_resolution))))

                        # x_um = (x_pos + 1) * local_resolution
                        # density -= max(0, ( 1 -
                        #              max(0, (x_um - start_min)
                        #                  / local_resolution)))
                        # # for the x bin in which the MT ended, don't add a
                        # # full MT but just the relative amount of the bin
                        # # crossed by the MT
                        # # x_um = x_pos * local_resolution
                        # density -= max(0, min(1,
                        #                       ((x_um - end) /
                        #                        local_resolution)))
                        length_in_range = density * local_resolution
                        # length_in_range = min(1,
                        #                    (end - x_pos *
                        #                     local_resolution)
                        #                    / local_resolution)
                        # length_in_range -= max(0, min(1,
                        #                            (start - (x_pos-1) *
                        #                             local_resolution) /
                        #                            local_resolution))
                        # if (sim_id == 2) & (core_id == 0):
                        #     print(density_sum, length_in_range,
                        #           x_um, start, end,
                        #           (x_um - start), (x_um - end),
                        #           local_density_here
                        #           )
                        density_sum += length_in_range
                        # subtract a maximum of 1 if the start
                        # is at the end of the range
                        if round(density_sum,6) >= round(threshold,6):
                            break
                        # if (length_in_range == 0) | (length_in_range < 0):
                        #     print(11223344, length_in_range,
                        #           x_pos, start_min, end)

                        # if ((round(density_sum, 6) < round(threshold, 6)) &
                        #         (core_id == 0) & (sim_id == 2)):
                        #     print(11223366, length_in_range,
                        #           x_pos*local_resolution, start_min, end,
                        #           density_sum, threshold, length_in_range,
                        #           local_density_here)
        object_position += 1

    if (round(density_sum, 6) < round(threshold, 6)):
        print(78910, sim_id, param_id, object_position,
              density_sum, threshold, length_in_range,
              local_density_here)
    # elif (core_id == 0) & (sim_id == 2):
    #     print(11111)

    return object_position, length_in_range, start

def _increase_lowest_no_object_idx(first_last_idx_with_object,
                                   object_states, sim_id, param_id):
    # if so, go forward from that position until the next empty idx is found
    object_pos = int(first_last_idx_with_object[0,sim_id, param_id])
    while object_pos < object_states.shape[1]:
        if object_states[0, object_pos, sim_id, param_id] == 0:
            break
        object_pos = object_pos + 1
    first_last_idx_with_object[0,sim_id, param_id] = object_pos

def _reduce_highest_object_idx(first_last_idx_with_object,
                               object_states, sim_id, param_id):
    # if so, go backward from that position until the next object is found
    object_pos = int(first_last_idx_with_object[1,sim_id, param_id])
    while object_pos > 0:
        if object_states[0, object_pos, sim_id, param_id] > 0:
            break
        object_pos = object_pos - 1
    first_last_idx_with_object[1,sim_id, param_id] = object_pos

def _remove_objects(all_object_removal_properties, object_removal_operations,
                    nb_objects_all_states, transition_parameters,
                    object_states, properties_array,
                    first_last_idx_with_object,
                    local_object_lifetime_array,
                                                 local_lifetime_resolution, times,
                    local_resolution,
                    nb_parallel_cores,  core_id, sim_id, param_id):
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

        nb_objects = first_last_idx_with_object[1,sim_id, param_id] + 1
        (object_pos,
         last_object_pos) = _get_first_and_last_object_pos(nb_objects,
                                                          nb_parallel_cores[sim_id, param_id],
                                                          core_id)
        while object_pos < last_object_pos:
            # combine property values according to property_operation
            combined_property_vals = properties_array[0, int(properties[0]),
                                                      int(object_pos),
                                                      sim_id, param_id]
            if math.isnan(combined_property_vals):
                object_pos += 1
            else:
                property_idx = 1
                while property_idx < properties.shape[0]:
                    property_nb = int(properties[property_idx])
                    property_val = properties_array[0, property_nb,
                                                    int(object_pos),
                                                    sim_id, param_id]
                    combined_property_vals += property_operation * property_val
                    property_idx += 1
                # check whether combined property values
                # are above or below threshold
                if threshold_operation == 1:
                    remove_object = combined_property_vals > threshold
                elif threshold_operation == -1:
                    remove_object = combined_property_vals < threshold

                # if object should be removed, set state to 0 and properties to NaN

                if remove_object:

                    object_state_to_remove = object_states[0, int(object_pos),
                                                           sim_id, param_id]

                    if nb_parallel_cores[sim_id, param_id] > 1:
                        cuda.atomic.add(nb_objects_all_states,
                                        (0, object_state_to_remove,
                                         sim_id, param_id),
                                        -1)
                        cuda.atomic.add(nb_objects_all_states,
                                        (0, 0, sim_id, param_id), 1)
                        cuda.atomic.min(first_last_idx_with_object,
                                        (0, sim_id, param_id), object_pos)
                        # if object_pos < first_last_idx_with_object[0,sim_id, param_id]:
                        #     first_last_idx_with_object[0,sim_id,
                        #                          param_id] = object_pos
                    else:
                        nb_objects_all_states[0, object_state_to_remove,
                                              sim_id, param_id] -= 1
                        # nb_objects_all_states[0, 0, sim_id, param_id] += 1
                        if object_pos < first_last_idx_with_object[0,sim_id,
                                                                   param_id]:
                            first_last_idx_with_object[0,sim_id,
                                                       param_id] = object_pos
                    # if object was generated by a resource limited generation
                    # transition, reduce number of objects generated from that
                    # transition and set object generation method to 0
                    if object_states[1, int(object_pos),
                                     sim_id, param_id] != 0:
                        nb_objects_all_states[1,
                                              int(transition_parameters[
                                                      object_states[
                                                          1, int(object_pos),
                                                          sim_id, param_id] - 1,
                                                      1]),
                                              sim_id, param_id] -= 1
                        # nb_objects_all_states[1,
                        #                       object_states[1,
                        #                                     int(object_pos),
                        #                                     sim_id, param_id]-1,
                        #                       sim_id, param_id] -= 1

                        object_states[1, int(object_pos),
                                      sim_id, param_id] = 0

                    if object_states[2, int(object_pos), sim_id, param_id] != 0:
                        object_states[2, int(object_pos), sim_id, param_id] = 0

                    object_states[0, int(object_pos),
                                  sim_id, param_id] = 0
                    property_nb = properties_array.shape[1] - 1
                    while property_nb > 0:
                        _track_object_property_changes(int(object_pos),
                                                       int(property_nb),
                                                       0,
                                                       properties_array,
                                                       local_object_lifetime_array,
                                                        local_lifetime_resolution,
                                                       times, sim_id, param_id)
                        properties_array[0, property_nb,
                                         int(object_pos),
                                         sim_id, param_id] = math.nan
                        property_nb -= 1
                object_pos += 1

        removal_nb += 1


def _save_values_with_temporal_resolution(timepoint_array, times,
                                         object_states, properties_array,
                                         time_resolution, start_save_time,
                                          save_initial_state,
                                          first_last_idx_with_object,
                                          nb_parallel_cores,  thread_masks,
                                          core_id, sim_id, param_id):

    if core_id == 0:
        current_timepoint = timepoint_array[0, sim_id, param_id]
        # check .if the next timepoint was reached, then save all values
        # at correct position
        time_jump = math.floor((times[0,sim_id, param_id] -
                                     (current_timepoint * time_resolution[0]))
                                    / time_resolution[0])
        current_timepoint += time_jump

        timepoint_array[0, sim_id, param_id] = current_timepoint

    if nb_parallel_cores[sim_id, param_id] > 1:
        cuda.syncwarp(thread_masks[0,sim_id, param_id])

    current_timepoint = timepoint_array[0, sim_id, param_id]
    # if the initial state was not saved then the index is actually the
    # timepoint minus 1
    if save_initial_state:
        timepoint_idx = int(current_timepoint)
    else:
        timepoint_idx = int(current_timepoint) - 1

    timepoint_idx -= max(0, int(math.floor(start_save_time /
                                           time_resolution[0])) - 1)

    # make sure that timepoint_idx is not higher than timepoint_array shape
    timepoint_idx = min(timepoint_idx, timepoint_array.shape[0] - 3)

    if core_id == 0:
        timepoint_array[timepoint_idx+2, sim_id, param_id] = current_timepoint

    # copy all current data into time-resolved data
    nb_objects = first_last_idx_with_object[1,sim_id, param_id] + 1
    (object_pos,
     last_object_pos) = _get_first_and_last_object_pos(nb_objects,
                                                      nb_parallel_cores[sim_id,
                                                                        param_id],
                                                      core_id)
    while object_pos < last_object_pos:
        if object_pos > 0:
            object_state = object_states[0, object_pos, sim_id, param_id]

            object_states[timepoint_idx+3,
                        object_pos,
                        sim_id, param_id] = object_state

            property_nb = 0
            while property_nb < properties_array.shape[1]:
                property_val = properties_array[0, property_nb,
                                                object_pos, sim_id, param_id]
                properties_array[timepoint_idx+1,
                                      property_nb,
                                      object_pos,
                                      sim_id, param_id] = property_val
                property_nb += 1
        object_pos += 1

    # no synchronization needed since no downstream processing
    # needs these synchronized changes, except the end of all simulations
    # which already include a synchronization step