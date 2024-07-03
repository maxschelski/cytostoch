import torch
import numpy as np
import time
import math
import numba
from numba import cuda
import functools
from . import simulation

from matplotlib import pyplot as plt

# @numba.cuda.jit
# def _get_first_and_last_object_pos(nb_objects, nb_parallel_cores, core_id):
#     objects_per_core = math.ceil(nb_objects / nb_parallel_cores)
#     if core_id < 0:
#         core_id_use = int(- (core_id + 1))
#     else:
#         core_id_use = int(core_id)
#
#     object_pos = objects_per_core * core_id_use
#     last_object_pos = objects_per_core * (core_id_use + 1)
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

@numba.cuda.jit
def get_hist_data(input_data, bins, bin_size, output, nb_operations,
                  timepoint,
                  first_last_idx_with_object):
    nb_parallel_cores = 1
    nb_processes = numba.cuda.gridsize(1)
    current_operation = numba.cuda.grid(1)
    nb_bins = bins.shape[0]
    while current_operation < nb_operations:
        param_nb = int(math.floor(current_operation /
                                  (input_data.shape[2] * nb_parallel_cores)))
        sim_nb = int(math.floor((current_operation -
                                  param_nb * input_data.shape[2] * nb_parallel_cores)
                                / nb_parallel_cores))
        # core_id = int(current_operation -
        #               param_nb * input_data.shape[2] * nb_parallel_cores -
        #               sim_nb * nb_parallel_cores)

        # (object_nb,
        #  last_object_nb) = _get_first_and_last_object_pos(first_last_idx_with_object[1,sim_nb, param_nb] + 1,
        #                                                   nb_parallel_cores,
        #                                                   core_id)
        object_nb = 0
        while object_nb < first_last_idx_with_object[1,sim_nb, param_nb] + 1:
            if not math.isnan(input_data[timepoint, object_nb, sim_nb, param_nb]):
                # go through each bin, with each bin being defined by the boundary
                # of the current bin_nb and the next bin_nb
                bin_nb = int(math.floor(input_data[timepoint, object_nb,
                                                   sim_nb, param_nb] / bin_size))
                bin_nb = min(bin_nb, nb_bins-1)
                # print(index, sim_nb, param_nb, bin_nb,
                #       input_data[0, index,
                #                  sim_nb, param_nb],
                #       bin_size, input_data.shape[0],
                #       input_data.shape[1], input_data.shape[2], input_data.shape[3])
                # numba.cuda.atomic.add(output, (0, bin_nb, sim_nb, param_nb), 1)
                # output[0, input_data[0, object_nb,
                #                      sim_nb, param_nb], sim_nb, param_nb] += 1
                output[timepoint, bin_nb, sim_nb, param_nb] += 1
                # bin_nb = 0
                # while bin_nb < (nb_bins - 1):
                #     if (bin_nb == (nb_bins - 2)) | (input_data[0, index, sim_nb, param_nb] < bins[bin_nb + 1]):
                #         # numba.cuda.atomic.add(output, (0, bin_nb, sim_nb, param_nb),
                #         #                       1)
                #         break
                #     bin_nb += 1
            object_nb += 1
        current_operation += nb_processes

@numba.cuda.jit
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

@numba.cuda.jit
def _get_local_and_total_density(local_density,
                                 local_resolution,
                                 start_nb_object, end_nb_object,
                                 position_array, length_array,
                                 timepoint, nb_parallel_cores):

    nb_processes = numba.cuda.gridsize(1)
    thread_id = numba.cuda.grid(1)
    grid = numba.cuda.cg.this_grid()

    # reassign_threads_time_step_size = 0.1
    # min_time_diff_for_reassigning = 0.1
    # nb_reassigned_threads = 1
    # rel_time_reassignment_check = 0.1

    total_nb_simulations = (position_array.shape[-2] *
                            position_array.shape[-1] *
                            nb_parallel_cores)
    # print(cuda.gridsize(1), total_nb_simulations)
    current_sim_nb = thread_id

    while current_sim_nb < total_nb_simulations:

        param_id = int(math.floor(current_sim_nb /
                                  (position_array.shape[-2] *
                                   nb_parallel_cores)))
        sim_id = int(math.floor((current_sim_nb -
                                 param_id * position_array.shape[-2] *
                                 nb_parallel_cores)
                                / nb_parallel_cores))
        core_id = int(current_sim_nb -
                      param_id * position_array.shape[-2] * nb_parallel_cores -
                      sim_id * nb_parallel_cores)

        (object_pos,
         last_object_pos) = _get_first_and_last_object_pos(end_nb_object -
                                                           start_nb_object,
                                                           nb_parallel_cores,
                                                           core_id)

        object_pos += start_nb_object
        while object_pos < (last_object_pos):
            if ((not math.isnan(length_array[timepoint, object_pos,
                                             sim_id, param_id]))):
                start = position_array[timepoint, object_pos,
                                             sim_id, param_id]
                # minimum allowed start position is negative local_resolution
                # since position 0 is the amount of MTs from
                # - local_resolution to 0
                # while the last position, is the amount of MTs from
                # x_max to local_resolution to x_max
                end = start + length_array[timepoint, object_pos,
                                           sim_id, param_id]
                # if not math.isnan(property_extreme_values[1, 0, 0]):
                #     end = min(property_extreme_values[1, 0, 0], end)
                start = max(start, 0)#-local_resolution)
                if (start != end) & (end >= 0):
                    x_start = int(math.floor(start / local_resolution))
                    x_end = int(math.floor(end / local_resolution))
                    x_pos = max(x_start, 0)
                    first_x = True
                    # track the cumulative density of this object
                    # to get the cumulative local density
                    while x_pos <= x_end:

                        density = 1
                        # if first_x:
                        #     # the start point is not actually at the very
                        #     # beginning of the resolution but after that
                        #     # therefore the added density is below 1 for the
                        #     # first position. Subtract the relative amount of
                        #     # the local resolution that the object starts
                        #     # after the x_pos
                        #     first_x = False
                        #     x_um = x_pos * local_resolution
                        #     density -= ((start - x_um) / local_resolution)
                        # if (x_pos == x_end):
                        #     # for the x bin in which the MT ended, don't add a
                        #     # full MT but just the relative amount of the bin
                        #     # crossed by the MT
                        #     x_um = x_pos * local_resolution
                        #     density -= ((x_um - end) / local_resolution)

                        cuda.atomic.add(local_density,
                                        (timepoint, x_pos, sim_id, param_id), density)
                        # density = 1
                        # if first_x | (x_pos == x_end):
                        #     # the start point is not actually at the very
                        #     # beginning of the resolution but after that
                        #     # therefore the added density is below 1 for the
                        #     # first position. Subtract the relative amount of
                        #     # the local resolution that the object starts
                        #     # after the x_pos
                        #     x_um = x_pos * local_resolution
                        #     density -= ((x_um - numba.cuda.selp(first_x,
                        #                                         start, end))
                        #                 / local_resolution)
                        #     first_x = False
                        #
                        #
                        # cuda.atomic.add(local_density,
                        #                 (timepoint, x_pos, sim_id, param_id), density)
                        x_pos += 1
            object_pos += 1
        current_sim_nb += nb_processes

def _run_operation_2D_to_1D_density_numba(sim_id, param_id, object_states, properties_array, times,
                                           transition_rates_array, all_transition_states,
                                           action_values, action_state_array,
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
                                           time_resolution, rng_states,
                                          execute_action_properties
                                           ):

    length_array = 0
    for length in dimensions[0].lengths:
        new_length_array = length.array
        # if regular_print:
        #     new_length_array = length.array
        # else:
        #     new_length_array = torch.concat(length.array_buffer)
        length_array = length_array + new_length_array

    if state_numbers is not None:
        # get mask for all objects in defined state
        object_states = simulation_object.object_states
        # if regular_print:
        #     object_states = simulation_object.object_states
        # else:
        #     object_states = torch.concat(simulation_object.object_states_buffer)
        mask = torch.zeros_like(object_states).to(torch.bool)
        for state in state_numbers:
            mask = (mask | (object_states == state))
        # get the maximum number of objects that should be analyzed
        # (from all simulations)
        # first sort mask so that first in the matrix there are all True
        # elements
        mask_sorted, idx = torch.sort(mask.to(torch.int), dim=1,
                                      stable=True, descending=True)
        # the first position that is not True is the sum of all True
        # elements across the first dim
        first_false_position = torch.count_nonzero(mask_sorted, dim=1)
        # then get the maximum of all simulations
        max_nb_objects = first_false_position.max()

        # set all properties of objects outside of mask to NaN
        # also use the maximum number of objects of interest for all
        # simulations, to discard all parts of the sorted array that only
        # contains objects which are not of interest
        position_array[mask_inv] = float("nan")
        position_array = torch.gather(position_array, dim=1, index=idx)
        position_array = position_array[:, :max_nb_objects]

        length_array[mask_inv] = float("nan")
        length_array = torch.gather(length_array, dim=1, index=idx)
        length_array = length_array[:, :max_nb_objects]

    # create boolean data array later by expanding each microtubule in space
    # size of array will be:
    # (max_position of neurite / resolution) x nb of microtubules
    min_position = dimensions[0].position.min_value
    if min_position is None:
        min_position = 0
    max_position = dimensions[0].position.max_value

    device = simulation_object.device

    positions = torch.arange(min_position, max_position + resolution * 0.9,
                             resolution).to(device)

    # print(position_array.shape, positions.shape)
    all_data = torch.zeros((position_array.shape[0], positions.shape[0],
                            *position_array.shape[2:])).to(device)

    # only if at least one element is True, analyze the data
    if mask.sum() > 0:

        position_dimension = int(round((max_position - min_position)
                                       / resolution, 5))
        # print(1, time.time() - start)
        start = time.time()
        # create tensors on correct device
        tensors = simulation_object.tensors

        # data type depends on dimension 0 - since that is the number of
        # different int values needed (int8 for <=256; int16 for >=256)
        # (dimension 0 is determined by max_x of neurite / resolution)
        if (position_dimension + 1) < 256:
            indices_tensor = torch.ByteTensor
            indices_dtype = torch.uint8
        else:
            indices_tensor = torch.ShortTensor
            indices_dtype = torch.short

        # extract positions of the array that actually contains objects
        # crop data so that positions that don't contain objects
        # are excluded
        # objects_array = ~torch.isnan(position_array)
        # positions_object = torch.nonzero(objects_array)
        # min_pos_with_object = positions_object[:,0].min()

        position_start = position_array
        # max_nb_objects = position_start.shape[0]
        # position_start = position_start[min_pos_with_object:max_nb_objects]

        # transform object properties into multiples of resolution
        # then transform length into end position
        position_start = torch.div(position_start, resolution,
                                   rounding_mode="floor")  # .to(torch.short)
        position_start = torch.unsqueeze(position_start, 1)

        position_end = length_array.unsqueeze(1)

        # position_end = position_end[min_pos_with_object:max_nb_objects]
        position_end = torch.div(position_end + position_array.unsqueeze(1),
                                 resolution,
                                 rounding_mode="floor")  # .to(indices_dtype)

        # # remove negative numbers to only look at inside the neurite
        # position_start[position_start < 0] = 0
        # position_start = position_start#.to(indices_dtype)

        # create indices array which each number
        # corresponding to one position in space (in dimension 0)
        indices = np.linspace(0, position_dimension, position_dimension + 1)
        indices = np.expand_dims(indices,
                                 tuple(range(1, len(position_start.shape) - 1)))
        indices = indices_tensor(indices).unsqueeze(0).to(device=device)

        # split by simulations to reduce memory usage, if needed
        # otherwise high memory usage leads to
        # massively increased processing time
        # with this split, processing time increases linearly with
        # array size (up to a certain max array size beyond which
        # the for loop leads to a supralinear increase in processing time)

        nb_objects = position_start.shape[2]
        # find way to dynamically determine ideal step size!
        step_size = nb_objects  # 5
        nb_steps = int(nb_objects / step_size)
        start_nb_objects = torch.linspace(0, nb_objects - step_size, nb_steps)

        # print(2, time.time() - start)
        start = time.time()
        for start_nb_object in start_nb_objects:
            end_nb_object = int((start_nb_object + step_size).item())
            start_nb_object = int(start_nb_object.item())
            # create boolean data array later by expanding each microtubule in space
            # use index array to set all positions in boolean data array to True
            # that are between start point and end point
            nb_timepoints = position_start.shape[0]
            data_array = ((indices.expand(nb_timepoints, -1, step_size,
                                          *position_start.shape[3:])
                           >= position_start[:, :,
                              start_nb_object:end_nb_object]) &
                          (indices.expand(nb_timepoints, -1, step_size,
                                          *position_start.shape[3:])
                           <= position_end[:, :,
                              start_nb_object:end_nb_object]))

            # then sum across microtubules to get number of MTs at each position
            data_array_sum = torch.sum(data_array, dim=2, dtype=torch.int16)

            all_data = all_data + data_array_sum
            # val_tmp = all_data[:2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    # print(3, time.time() - start)
    start = time.time()
    data_array = all_data[:, :-1]

    dimensions = [-1] + [1] * (len(data_array.shape) - 2)
    positions = positions.view(*dimensions)[:-1]
    positions = positions.unsqueeze(0).expand([data_array.shape[0],
                                               *positions.shape])


class PropertyGeometry():

    def __init__(self, properties, operation):
        """
        Define geometries of object properties by letting min or max values
        of one property depend on other properties

        Args:
            properties (list of ObjectProperty objects): List of properties that
                defines the input to the operation function. For currently
                implemented operations, the first property must have the
                max_value defined as number (int or float).
            operation (function or string): function that has 'properties'
                as input and outputs per position allowed value as flattened
                array  (can be min or max, depending on the ObjectProperty
                parameter it is used for);
                as string only for implemented  functions:
                - "same_dimension_forward": Only works for one property in
                "properties"; the property for which the geometry is set and the
                "properties" in PropertyGeometry are in the same dimension,
                implemented for forward movement (towards higher values),
                which means the difference of the current position of
                "properties" to it's maximum position is the maximum possible
                value
        """
        implemented_operations = {}
        new_operation = self.operation_same_dimension_forward
        implemented_operations["same_dimension_forward"] = new_operation
        self.properties = properties
        self._operation = operation
        if type(operation) == str:
            if operation not in implemented_operations:
                raise ValueError(f"The supplied PropertyGeometry operation "
                                 f"{operation} is not implemented. Only the "
                                 f"following operations are implemented:"
                                 f"{', '.join(implemented_operations.keys())}")
            self.operation = implemented_operations[operation]
        else:
            self.operation = operation

    def operation_same_dimension_forward(self, properties):
        """

        Args:
            properties (list): List of ObjectProperty objects. First object
                must be object that determines the max value

        Returns:

        """
        # if len(properties) > 1:
        #     raise ValueError(f"The PropertyGeometry operation "
        #                      f"'same_dimension_forward' is only implemented for "
        #                      f"one 'ObjectProperty' object in the 'properties' "
        #                      f"parameter. Instead {len(properties)} were "
        #                      f"supplied.")
        if type(properties[0].max_value) not in [float, int]:
            raise ValueError(f"For the PropertyGeometry operation "
                             f"'same_dimension_forward', only an ObjectProperty"
                             f" in the parameter 'properties' that has the "
                             f"max_value defined as float or int is allowed. "
                             f"Instead the max_value is "
                             f"{properties[0].max_value}.")
        if properties[0].closed_max:
            max_values = properties[0].max_value - properties[0].array

            # subtract values of each property in the same dimension from max value
            # to get
            if len(properties) > 1:
                for property in properties[1:]:
                    max_values -= property.array
        else:
            max_values = math.nan

        return max_values
        # else:
        #     return None

    def get_limit(self):
        return self.operation(self.properties)

class Dimension():

    def __init__(self, positions, lengths=None, direction=None):
        """

        Args:
            positions (list of ObjectProperty objects): First value in list is
                the starting position that also contains possible
                min and max values. Further values in list correspond to
                values added to position for quantifications.
            length (list): list of ObjectProperty objects that together define
                the object length
            direction (ObjectProperty):
        """
        self.positions = positions
        self.lengths = lengths
        self.direction = direction

class DataExtraction():
    """
    Compress data for keeping and saving.
    Define which properties should be kept and how they should be rearranged.
    So far, this is only implemented for a 1D system modeled in 2D
    (object start position and object length).
    """

    def __init__(self, dimensions, operation, state_groups=None,
                 print_regularly=False, show_data=True, **kwargs):
        """
        Define how many dimensions in the system

        Args:
            dimensions (list of Dimension objects):
            operation (func or string): operation used to transform
                data for extraction. Name of already implemented function as
                string. Alternatively, user-defined function (encouraged to
                submit pull request to add to package) using parameters
                dimensions and resolution.
            extract_state (bool): Whether to extract object state information
            state_groups (dict of lists with state objects): dict with each
                value being a state groups, which is a list in which each
                entry is a State object. The keys are the names for the state
                group. Each group will be analyzed as one entity.
                (e.g. for [[1,2],[3]], where each number is a State object,
                state 1 and 2 will be analyzed as one state but state 3
                will be analyzed as a separate state)
            regular_print (bool): Whether the extraction is part of the regular
                print for monitoring and therefore should only contain data from
                the last iteration
        """
        self.dimensions = dimensions
        self.kwargs = kwargs
        self.state_groups = state_groups
        self.print_regularly = print_regularly
        self.operation_name = operation
        self.show_data = show_data

        self.resolution = kwargs.get("resolution", None)

        implemented_operations = {}
        new_operation = self._operation_2D_to_1D_density
        implemented_operations["2D_to_1D_density"] = new_operation
        implemented_operations["raw"] = self._operation_raw
        implemented_operations["global"] = self._operation_global
        implemented_operations["length_distribution"] = self._length_distribution

        if type(operation) == str:
            if operation not in implemented_operations:
                raise ValueError(f"For DataExtraction only the following"
                                 f" operations are implemented and can be "
                                 f"refered to by name in the 'operation' "
                                 f"parameter: "
                                 f"{', '.join(implemented_operations.keys())}."
                                 f" Instead the following name was supplied: "
                                 f"{operation}.")
            self.operation = implemented_operations[operation]
        else:
            self.operation = operation

    def extract(self, simulation_object, regular_print=False, **kwargs):

        all_kwargs = {**self.kwargs, **kwargs}

        if self.state_groups is None:
            data, _ = self.operation(self.dimensions, simulation_object,
                                     regular_print=regular_print,
                                **all_kwargs)
            all_data = data
        elif type(self.state_groups) == str:
            if self.state_groups.lower() != "all":
                raise ValueError("The only allowed string value for state "
                                 "groups is 'all'. Instead the value was:"
                                 f" {self.state_groups}.")
            all_data = {}
            for state in simulation_object.states:
                data, data_col_names = self.operation(self.dimensions,
                                                  simulation_object,
                                                  state_numbers=
                                                  [state.number],
                                                      regular_print=
                                                      regular_print,
                                                  **all_kwargs)
                # extract the actual data and not auxiliary information
                # which would be the same between different state groups
                # due to same underlying space structure
                for name, data_vals in data.items():
                    # data_col_names contains all names of the actual data
                    # which is different between state groups
                    if name in data_col_names:
                        all_data[name+"_"+state.name] = data_vals
                    else:
                        # add auxiliary information once
                        if name not in all_data.keys():
                            all_data[name] = data_vals
        else:
            all_data = {}
            for group_name, state_group in self.state_groups.items():
                state_numbers = [state.number for state in state_group]
                # state_numbers_str = [str(state) for state in state_numbers]
                # state_string = "S"+"-".join(state_numbers_str)
                data, data_col_names = self.operation(self.dimensions,
                                                 simulation_object,
                                                  state_numbers=
                                                  state_numbers,
                                                      regular_print=
                                                      regular_print,
                                                 **all_kwargs)
                # extract the actual data and not auxiliary information
                # which would be the same between different state groups
                # due to same underlying space structure
                for name, data_vals in data.items():
                    # data_col_names contains all names of the actual data
                    # which is different between state groups
                    if name in data_col_names:
                        all_data[name+ "_" + group_name] = data_vals
                    else:
                        # add auxiliary information once
                        if name not in all_data.keys():
                            all_data[name] = data_vals

        return all_data


    def _length_distribution(self, dimensions, simulation_object, properties, state_numbers, max_length,
                             first_last_idx_with_object,
                             bin_size=None, nb_bins=None,
                             **kwargs):

        start = time.time()
        object_states = simulation_object.object_states

        # get mask of each position not being in any of the state_numbers
        mask = torch.zeros_like(torch.Tensor(object_states)).to(torch.bool)
        mask_inv = torch.full(object_states.shape, 1).to(torch.bool)
        for state in state_numbers:
            mask_inv = (mask_inv & (object_states != state))
            mask = (mask | (object_states == state))

        # convert list or tuple of properties to dict by using property names
        # as key
        property_dict = {}
        for property_name, property in properties.items():
            property_dict[property_name] = property

        # get data of properties from buffer (which contains data from multiple
        # iterations/timepoints). Also sum up multiple properties, if defined.
        analysis_properties = {}
        for name, property in property_dict.items():
            if type(property) in [list, tuple]:
                property_array = torch.zeros_like(property[0].array)
                for sub_property in property:
                    sub_property_array = sub_property.array
                    # if regular_print:
                    #     sub_property_array = sub_property.array.unsqueeze(0)
                    # else:
                    #     sub_property_array = torch.concat(sub_property.array_buffer)
                    property_array = property_array + sub_property_array
                    # property_array = torch.nansum(torch.concat([property_array.unsqueeze(0),
                    #                                             sub_property_array.unsqueeze(0)],
                    #                                            dim=0),dim=0)
            else:
                property_array = property.array.clone()
                # if regular_print:
                #     property_array = property.array.unsqueeze(0)
                # else:
                #     property_array = torch.concat(property.array_buffer)

            property_array[mask_inv] = float("nan")

            position = dimensions[0].positions[0].array.clone()
            # only consider length starting from start of neurite
            # therefore add all negative positions to property array
            position[position > 0] = 0

            analysis_properties[name] = property_array + position

        # create histogram of distributions for

        data_dict = {}
        if nb_bins is None:
            nb_bins = int(np.ceil(max_length / bin_size)) + 1
        bins = np.linspace(0, max_length, nb_bins).astype(np.float32)

        # bins_torch = torch.linspace(0, max_length, nb_bins).to(torch.float32).to(simulation_object.device)

        bins_cuda = numba.cuda.to_device(np.ascontiguousarray(bins))
        all_data_columns = ["number"]
        first_last_idx_with_object = numba.cuda.to_device(np.ascontiguousarray(first_last_idx_with_object))
        nb_SM, nb_cc = simulation.SSA._get_number_of_cuda_cores()

        for name, values in analysis_properties.items():
            #all_state_values = values[mask].cpu()

            # value_hist = np.histogram(all_state_values, bins=nb_bins, range=(0, max_length))
            # bins = torch.linspace(0, max_length, nb_bins)
            # bins = bins.to(simulation_object.device)

            # get bin number for each value
            # values_bin = torch.floor(values / bin_size).to(torch.int32)
            # print(values_bin)
            # values_bin = torch.bucketize(values, boundaries=bins_torch)
            # print(values_bin)
            # print(values)

            hist = np.zeros((values.shape[0], nb_bins-1, *values.shape[2:]))
            hist = numba.cuda.to_device(np.ascontiguousarray(hist))

            nb_operations = values.shape[-1] * values.shape[-2]

            values_cuda = numba.cuda.to_device(np.ascontiguousarray(np.array(values.cpu())))
            start = time.time()
            for timepoint in range(object_states.shape[0]):
                get_hist_data[nb_SM, nb_cc](values_cuda, bins_cuda, bin_size, hist,
                                            nb_operations, timepoint,
                                            first_last_idx_with_object)
            # print("length extr time: ", time.time() - start)
            numba.cuda.synchronize()
            hist = hist.copy_to_host()

            # test_hist = np.digitize(values.cpu(), bins=bins)
            # # test_hist = torch.bucketize(values, boundaries=bins)
            # def get_hist_data_np(data, nb_bins):
            #     values, count = np.unique(data, return_counts=True)
            #     # values, count = torch.unique(data, return_counts=True)
            #     hist = np.zeros(nb_bins)
            #     # hist = torch.zeros(nb_bins)
            #     hist[values-1] = count
            #     return hist
            #
            # from functools import partial
            #
            # get_hist_data_bins = partial(get_hist_data_np, nb_bins = nb_bins)
            # hist_np = np.apply_along_axis(get_hist_data_bins, axis=1, arr=test_hist)[:,:-1]

            # use numpy.histogram to compare results to custom implementation
            # print(np.histogram(values.cpu(), bins=nb_bins, range=(0, max_length))[0]/values.cpu().shape[2])

            bins[-2] += bins[-1]
            bins = bins[:-1]

            data_dict["length_hist_" + name+"_bins"] = torch.Tensor(bins).unsqueeze(0).unsqueeze(2).unsqueeze(3).expand(hist.shape)
            data_dict["length_hist_" + name+"_numbers"] = torch.Tensor(hist)
            all_data_columns.append("length_hist_" + name+"_bins")
            all_data_columns.append("length_hist_" + name+"_numbers")

        return data_dict, all_data_columns

    def _operation_global(self, dimensions, simulation_object, properties,
                                     state_numbers, regular_print, **kwargs):
        """
        Get global properties of object in states.
        Args:
            properties (dict or list with ObjectProperty objects): Dict with
                values being properties to be analyzed separately. Each value
                can either be an ObjectProperty or a list of ObjectProperty
                objects that will be summed together.
                Key is the name of the property (string). If list, the name
                of the ObjectProperty will be used as key and only one property
                can be used per group (no nested lists allowed).
            state_numbers (list of State objects): List of state objects that
                should be analyzed as one entity.

        Returns:

        """
        object_states = simulation_object.object_states

        # if regular_print:
        #     object_states = simulation_object.object_states.unsqueeze(0)
        # else:
        #     object_states = torch.concat(simulation_object.object_states_buffer)

        mask = torch.zeros_like(torch.Tensor(object_states)).to(torch.bool)
        mask_inv = torch.full(object_states.shape, 1).to(torch.bool)
        for state in state_numbers:
            mask = (mask | (object_states == state))
            mask_inv = (mask_inv & (object_states != state))

        # convert list or tuple of properties to dict by using property names
        # as key
        property_dict = {}
        if type(properties) in [list, tuple]:
            for property in properties:
                if type(property) in [list, tuple]:
                    raise ValueError("If the 'properties' parameter for the "
                                     "data extraction operation 'global' is "
                                     "a list, then no element in that list can "
                                     "be an iterable and must be an "
                                     "ObjectPropery object.")
                property_dict[property.name] = property
        else:
            property_dict = properties

        # get data of properties from buffer (which contains data from multiple
        # iterations/timepoints). Also sum up multiple properties, if defined.
        analysis_properties = {}
        for name, property in property_dict.items():
            if type(property) in [list, tuple]:
                property_array = 0
                for sub_property in property:
                    sub_property_array = sub_property.array
                    # if regular_print:
                    #     sub_property_array = sub_property.array.unsqueeze(0)
                    # else:
                    #     sub_property_array = torch.concat(sub_property.array_buffer)
                    property_array = property_array + sub_property_array
            else:
                property_array = property.array.clone()
                # if regular_print:
                #     property_array = property.array.unsqueeze(0)
                # else:
                #     property_array = torch.concat(property.array_buffer)

            property_array[mask_inv] = float("nan")
            analysis_properties[name] = property_array

        data_dict = {}
        object_number = mask.sum(dim=1).unsqueeze(1)
        data_dict["number"] = object_number.cpu()

        all_data_columns = ["number"]
        for name, values in analysis_properties.items():
            mean_values = values.nanmean(dim=1).unsqueeze(1)
            data_dict["mean_"+name] = mean_values.cpu()
            # get mean data for objects completely inside (position >= 0)
            values[dimensions[0].positions[0].array < 0] = math.nan
            inside_mean_values = values.nanmean(dim=1).unsqueeze(1)
            data_dict["mean_inside_"+name] = inside_mean_values.cpu()
            data_dict["mass_"+name] = (mean_values.cpu() * object_number.cpu())
            all_data_columns.append("mean_"+name)
            all_data_columns.append("mean_inside_"+name)
            all_data_columns.append("mass_"+name)

        return data_dict, all_data_columns

    def _operation_raw(self, dimensions, **kwargs):
        data_dict = {}
        data_dict["position"] = dimensions[0].positions[0].array.cpu()
        data_dict["length"] = dimensions[0].length.array.cpu()
        return data_dict


    def _operation_2D_to_1D_density(self, dimensions, simulation_object,
                                    state_numbers=None, regular_print=False,
                                    resolution=0.2, end_density=False,
                                    **kwargs):
        """
        Create 1D density array from start and length information without
        direction.
        Args:
            dimensions (list of Dimension objects):
            resolution (float): resolution in dimension for data export
            state_numbers (list of ints): State numbers to analyze

        Returns:

        """
        if len(dimensions) > 1:
            return ValueError(f"The operation '2D_to_1D_density' is only "
                              f"implemented for 1 dimension. DataExtraction "
                              f"received {len(dimensions)} dimensions instead.")

        position_array = 0
        for position in dimensions[0].positions:
            position_array = position_array + position.array

        # if regular_print:
        #     position_array = dimensions[0].position.array
        # else:
        #     position_array = torch.concat(dimensions[0].position.array_buffer)

        length_array = 0
        if not end_density:
            for length in dimensions[0].lengths:
                new_length_array = length.array
                # if regular_print:
                #     new_length_array = length.array
                # else:
                #     new_length_array = torch.concat(length.array_buffer)
                length_array = length_array + new_length_array
        else:
            length_array = position_array.clone()
            length_array[:] = 0

        if state_numbers is not None:
            # get mask for all objects in defined state
            object_states = simulation_object.object_states
            # if regular_print:
            #     object_states = simulation_object.object_states
            # else:
            #     object_states = torch.concat(simulation_object.object_states_buffer)
            mask = torch.zeros_like(object_states).to(torch.bool)
            mask_inv = torch.full(object_states.shape, 1).to(torch.bool)
            for state in state_numbers:
                mask = (mask | (object_states == state))
                mask_inv = (mask_inv & (object_states != state))

            # get the maximum number of objects that should be analyzed
            # (from all simulations)
            # first sort mask so that first in the matrix there are all True
            # elements
            mask_sorted, idx = torch.sort(mask.to(torch.int), dim=1,
                                          stable=True, descending=True)
            # the first position that is not True is the sum of all True
            # elements across the first dim
            first_false_position = torch.count_nonzero(mask_sorted, dim=1)
            # then get the maximum of all simulations
            max_nb_objects = first_false_position.max()

            idx = idx.to("cuda")

            # set all properties of objects outside of mask to NaN
            # also use the maximum number of objects of interest for all
            # simulations, to discard all parts of the sorted array that only
            # contains objects which are not of interest
            position_array[mask_inv] = float("nan")
            position_array = torch.gather(position_array, dim=1,
                                          index=idx)
            position_array = position_array[:, :max_nb_objects]

            length_array[mask_inv] = float("nan")
            length_array = torch.gather(length_array, dim=1, index=idx)
            length_array = length_array[:, :max_nb_objects]

        # only if at least one element is True, analyze the data
        if mask.sum() > 0:

            # create boolean data array later by expanding each microtubule in space
            # size of array will be:
            # (max_position of neurite / resolution) x nb of microtubules
            min_position = dimensions[0].positions[0].min_value
            if min_position is None:
                min_position = 0
            max_position = dimensions[0].positions[0].max_value

            position_dimension = int(round((max_position - min_position)
                                           / resolution,5)) + 1

            positions = torch.arange(min_position,
                                     max_position + resolution * 0.9,
                                     resolution)

            nb_objects = length_array.shape[1]
            # find way to dynamically determine ideal step size!
            step_size = nb_objects  # 5
            nb_steps = int(nb_objects / step_size)
            start_nb_objects = torch.linspace(0, nb_objects - step_size,
                                              nb_steps)

            all_data = 0

            for start_nb_object in start_nb_objects:
                end_nb_object = int((start_nb_object + step_size).item())
                end_nb_object = min(end_nb_object, nb_objects)
                start_nb_object = int(start_nb_object.item())

                to_cuda = lambda x:numba.cuda.to_device(np.ascontiguousarray(x))

                local_density = np.zeros((object_states.shape[0],
                                          position_dimension,
                                          *object_states.shape[-2:]))
                local_density = to_cuda(local_density)

                position_array_cuda = to_cuda(position_array[:,
                                              start_nb_object:
                                              end_nb_object].cpu().to(
                    torch.float32))

                length_array_cuda = to_cuda(length_array[:,start_nb_object:
                                                           end_nb_object].cpu())

                nb_parallel_cores = 32

                nb_SM, nb_cc = simulation.SSA._get_number_of_cuda_cores()

                for timepoint in range(object_states.shape[0]):
                    _get_local_and_total_density[nb_SM, nb_cc](local_density,
                                                               resolution,
                                                               start_nb_object,
                                                               end_nb_object,
                                                               position_array_cuda,
                                                               length_array_cuda,
                                                               timepoint,
                                                               nb_parallel_cores)

                local_density = local_density.copy_to_host()
                data_array_sum = local_density
                numba.cuda.synchronize()

                # # then sum across microtubules to get number of MTs at each position
                # data_array_sum = torch.sum(data_array, dim=2, dtype=torch.int16)
                all_data = all_data + data_array_sum
                # val_tmp = all_data[:2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                del data_array_sum
                torch.cuda.empty_cache()

        data_array = all_data[:,:]

        dimensions = [-1] + [1] * (len(data_array.shape) - 2)
        positions = positions.view(*dimensions)[:-1]
        positions = positions.unsqueeze(0).expand([data_array.shape[0],
                                                   *positions.shape])

        data_dict = {}
        data_dict["1D_density_position"] = positions
        data_dict["1D_density"] = torch.Tensor(data_array)
        del positions

        torch.cuda.empty_cache()

        return data_dict, ["1D_density"]



class ObjectProperty():

    def __init__(self, min_value=None, max_value=None,
                 start_value=[0,1],
                 initial_condition=None, name=""):
        """
        Args:
            min_value (float or PropertyGeometry): lower limit on property
                value. Defines geometry of physical space.
            max_value (float or PropertyGeometry): upper limit on property
                value. Deinfes geometry of physical space.
            start_value (float or PropertyGeometry or list of two values/
                PropertyGeometry):
                Value at which new objects will be initialized, if list, will
                be initialized randomly between first (min_value) and second
                (max_value) element.
            initial_condition (float or PropertyGeometry, or list with
                2 floats/PropertyGeometry or func):
                Define the values used at the  beginning of each
                simulation. If list, will be initialized randomly between first
                (min_value) and second (max_value) element.
                Can't be None when start_value is None. If None, will be
                assigned according to start_value.
                If function, should take the number of events and output
                1D array with corresponding values. So far object property
                initial conditions are independent of the object state.
            name (String): Name of property, used for data export
                and readability.
        """
        self._min_value = min_value
        self._max_value = max_value
        self._start_value = start_value
        self.closed_min = True
        self.closed_max = True
        self._initial_condition = initial_condition
        self.name = "prop"
        if (name != "") & (name is not None):
            self.name += "_" + name

        if (start_value is None):
            raise ValueError("Start_value has to be float or a list of two "
                             "floats ([min, max]), to know at which "
                             "property value to initialize new objects. For "
                             "the object property {name}, it start_value is "
                             "None instead.")

        if type(start_value) == list:
            if len(start_value) != 2:
                raise ValueError("Start_value for object property can only be "
                                 "a float or a list with two elements. For the "
                                 f"object property {name}, start_value is a "
                                 f"list with {len(start_value)} elements "
                                 f"instead.")

        # initialize variables used in simulation
        self.array = torch.Tensor([])
        self.array_buffer = []

        self.saved = None

    @property
    def min_value(self):
        if type(self._min_value) == type(PropertyGeometry):
            return self._min_value.get_limit()
        return self._min_value

    @property
    def max_value(self):
        if type(self._max_value) == type(PropertyGeometry([],[])):
            return self._max_value.get_limit()
        return self._max_value

    @max_value.setter
    def max_value(self, value):
        self._max_value = value

    @property
    def start_value(self):
        if type(self._start_value) == list:
            current_start_value = []
            for value in self._start_value:
                if type(value) == type(PropertyGeometry):
                    current_start_value.append(value.get_limit())
                else:
                    current_start_value.append(value)
        else:
            current_start_value = self._start_value
        return current_start_value

    @property
    def initial_condition(self):
        if type(self._initial_condition) == list:
            current_initial_condition = []
            for value in self._initial_condition:
                if type(value) == "PropertyGeometry":
                    current_initial_condition.append(value.get_limit())
                else:
                    current_initial_condition.append(value)
        else:
            current_initial_condition = self._initial_condition
        return current_initial_condition

class ObjectPosition(ObjectProperty):
    def __init__(self, min_value=None, closed_min=True, max_value=None,
                 closed_max=True, start_value=[0,1],
                 initial_condition=None, name=""):
        """

        Args:
            min_value (float or PropertyGeometry): lower limit on property
                value. Defines geometry of physical space.
            closed_min (Bool): Whether objects can't go below the min_value
                (flooring)
            max_value (float or PropertyGeometry): upper limit on property
                value. Deinfes geometry of physical space.
            closed_max (Bool): Whether objects can't go above the max_value
                (ceiling)
            start_value (float or PropertyGeometry or list of two values/
                PropertyGeometry):
                Value at which new objects will be initialized, if list, will
                be initialized randomly between first (min_value) and second
                (max_value) element.
            initial_condition (float or PropertyGeometry, or list with
                2 floats/PropertyGeometry or func):
                Define the values used at the  beginning of each
                simulation. If list, will be initialized randomly between first
                (min_value) and second (max_value) element.
                Can't be None when start_value is None. If None, will be
                assigned according to start_value.
                If function, should take the number of events and output
                1D array with corresponding values. So far object property
                initial conditions are independent of the object state.
            name (String): Name of property, used for data export
                and readability.
        """
        super().__init__(min_value, max_value, start_value, initial_condition,
                         name)
        self.closed_min = closed_min
        self.closed_max = closed_max


class Action():

    def __init__(self, object_property, operation, parameter, states = None,
                 name=""):
        """

        Args:
            object_property (ObjectProperty object):
            operation (func or str): function that takes the property tensor, the
                action values tensor and the time tensor as multi-D tensor and
                 a mask that is True at all positions where the action will
                 be executed, and False at positions where it won't be executed
                 and then outputs the transformed property tensor.
                Alternatively use one of the following standard_positions as
                string: "add", "subtract"
            parameter (Parameter object): Parameter that contains values used
                for the action.
            states (list of State objects): States in which action will be
                executed. If None, will be executed on all non zero states
            name (String): Name of action, used for data export and readability
        """
        self.states = states
        self.parameter = parameter
        self.object_property = object_property
        if name != "":
            self.name = "Act_"+name
        else:
            self.name = "Act"
            # if states is not None:
            #     self.name += "_S"+str(states)
            if ((object_property.name != "property") &
                    (object_property.name is not None)):
                self.name += "_"+object_property.name
            self.name += "_OP"+str(operation)

        implemented_operations = {}
        implemented_operations["add"] = self._operation_add
        implemented_operations["subtract"] = self._operation_subtract

        self._operation = operation

        # FINISH TRANSFERING OPERATION FROM SIMULATION TO ACTION CLASS
        # ONLY EXECUTE OPERATION FROM SIMULATION
        # FIX BUGS
        error_msg = (f"The action operation {operation}"
                     f" is not defined. ")
        error_details = (f"Please either choose one of the "
                         f"following function names as string: "
                         f"{', '.join(list(implemented_operations.keys()))}."
                         f" Or alternatively define a function"
                         f" for the 'operation' parameter that takes"
                         f" the property_array, reaction_times and the"
                         f" value array of the action as parameters.")
        # choose a function of one of the established functions
        if type(operation) == str:
            if operation not in implemented_operations:
                raise ValueError(error_msg + error_details)
            self.operation = implemented_operations[operation]

        elif type(operation) != type(self.__init__):
            raise ValueError("Only strings or functions are allowed." +
                             error_details)
        else:
            self.operation = operation

    def _operation_add(self, property_values, values, time, mask):
        return property_values + values*time*mask

    def _operation_subtract(self, property_values, values, time, mask):
        return property_values - values*time*mask


class State():

    def __init__(self, initial_condition=0, name=""):
        """

        Args:
            name (str): name of state, used for data export and readability
            initial_condition (int): Number of objects in that state
        """
        self.initial_condition = initial_condition
        self.name = "state"
        if (name != "") & (name is not None):
            self.name += "_" + name

        if type(self.initial_condition) is not list:
            self.initial_condition = np.array([self.initial_condition])

        # set variable to treat initial condtions as other simulation parameters
        self.values = self.initial_condition

class ObjectRemovalCondition():
    """
    Define conditions to remove objects from simulations - e.g. if they move
    out of the region of interest. In contrast, min and max values of properties
    prevent objects from moving out.
    """

    def __init__(self, object_properties, combine_properties_operation,
                 compare_to_threshold, threshold):

        """

        Args:
            object_properties (list of ObjectProperties):
            operation (function or string): If function, has to take parameters:
                object_properties and threshold. It should output a mask of
                objects that should be removed
            threshold (float): Value of last not removed object for operation
        """
        self.object_properties = object_properties
        self.combine_properties_operation = combine_properties_operation
        self.compare_to_threshold = compare_to_threshold
        # implemented_operations = {}
        # new_operation = self.operation_sum_smaller_than
        # implemented_operations["sum_smaller_than"] = new_operation
        implemented_property_operations = {}
        implemented_property_operations["sum"] = np.sum
        implemented_property_operations["subtract"] = lambda x: x[0] - np.sum(x[1:])

        implemented_comparison_ooperations = {}
        implemented_comparison_ooperations["smaller"] = (lambda x, y:
                                                         x < threshold)
        implemented_comparison_ooperations["larger"] = (lambda x, y:
                                                         x > threshold)

        implemented_operations = {"combine_properties":
                                      implemented_property_operations,
                                  "compare": implemented_comparison_ooperations}

        operations = {"combine_properties": combine_properties_operation,
                      "compare": compare_to_threshold}
        self.operations = {}
        for operation_name, operation in operations.items():
            if type(operation) == str:
                if (operation
                        not in implemented_operations[operation_name]):
                    operation_strings = implemented_operations[operation_name].keys()
                    raise ValueError(f"For ObjectRemovalCondition only the following"
                                     f" operations are implemented and can be "
                                     f"refered to by name in the 'operation' "
                                     f"paramter: "
                                     f"{', '.join(operation_strings)}."
                                     f" Instead the following name was supplied: "
                                     f"{operation}.")
                operation = implemented_operations[operation_name][operation]
                self.operations[operation_name] = operation
            else:
                self.operations[operation_name] = operation
        self.threshold = threshold

    # def get_objects_to_remove(self):

    # @numba.njit
    # @staticmethod
    # def operation_sum_smaller_than(object_properties, threshold):
    #     sum_of_properties = np.zeros_like(object_properties[0].array)
    #     for object_property in object_properties:
    #         sum_of_properties += object_property.array
    #     return sum_of_properties < threshold

    # def get_objects_to_remove_func(self):
    #     return functools.partial(self.operations[],
    #                              threshold=self.threshold)

class Parameter():

    def __init__(self, values, scale="rates", convert_half_lifes=True,
                 dependence=None, switch_timepoints=None,
                 name=""):
        """

        Args:
            values (Iterable): 1D Iterable (list, numpy array) of all values
                which should be used. If scale is half-lifes, then values will
                be converted to rates if convert_half_lifes is True.
                Each sublist of parameter values represents the values until
                a defined switchpoint. The index of the parameter values indicates
                the index of the switch_timepoints at which parameter values
                are switched to the next index (next group). For the last group
                there is no switch timepoint defined, since these parameter
                values will be active until the last timepoint.
            scale (String): Type of values supplied, can be "rates", "half-lifes"
                or "other". For half-lifes, values will be converted to rates
                if convert_half_lifes is True.
            convert_half_lifes: Whether to convert half lifes to rates
            dependence: Dependence object to make parameter value
                depend on something else,
                e.g. position with PropertyDependence
            switch_timepoints: Timepoints (in minutes) at which parameter values
                change to the values in the next sublist. The number of switch
                timepoints has to equal the number of sublists minus 1. E.g.
                if two sublists of parameter values are defined
                (e.g. [[0,1,2],[1,2,3]]) then one switch timepoint should be
                defined.
            name (string): Name of parameter
        """
        if type(values[0]) not in [list, tuple]:
            values = [values]

        if (scale == "half-lifes") & (convert_half_lifes):
            lifetime_to_rates_factor = np.log(np.array([2]))
            self.values = torch.DoubleTensor(lifetime_to_rates_factor / values)
        else:
            self.values = torch.DoubleTensor(values)

        self.name = name
        self.number = None
        self.value_array = torch.DoubleTensor([])
        self.dependence = dependence
        self.switch_timepoints = None
        if switch_timepoints is not None:
            self.switch_timepoints = np.array(switch_timepoints)

class PropertyDependence():

    """
    Define a dependence of a parameter on the position of the object.
    """

    def __init__(self, start_val = None, end_val = None,
                 param_change = None,
                 param_change_is_abs = False,
                 prop_change_is_abs = False,
                 properties = None,
                 function="linear",
                 name=None):
        """
         implement parameters depending on property values,
        as added to normal/baseline parameter value
        as two values: value at start and/or end of neurite,
        if both are defined, then the change per um is calculated
        and definitions of change per um etc are not taken into
        account
        if one of the values is defined then the change from that
        position (start or end of neurite) is defined by:
        absolute or relative change per um or per % of length
        Thereby, the added linearly changing part might even go to 0

        Params:
            start_val: Parameter object for absolute start value of parameter.
            end_val: Parameter object for absolute end_val of parameter.
                If end_val is defined but start_val is not defined, then
            param_change: Parameter object for
                change of parameter per position change
            param_change_is_abs: Boolean of whether
                change of parameter is absolute
                (in parameter units) or relative (in fraction of difference
                between end_val and start_val)
            prop_change_is_abs: Boolean of whether
                change of position in param_change
                is absolute (in um) or relative (fraction of length)
            properties: List of ObjectProperty objects that should be summed
                to obtain the value that the parameter depends on
            function: String "linear" or "exponential", determining the
                type of parameter dependence. If "exponential", the param_change
                is the rate (lambda) of the exponential function.
        """
        if (start_val is None) & (end_val is None):
            error_msg = ("When defining a dependence, the start "
                         "and/or the end value have to be defined. "
                         "Instead, both were None.")
            if name is not None:
                error_msg += f" For the dependence {name}."
            raise ValueError(error_msg)
        self.start_val = start_val

        self.end_val = end_val

        if ((start_val is not None) & (end_val is not None) &
                (param_change is not None)):
            raise ValueError("When defining a dependence, not all of "
                             "start_val, end_val and param_change "
                             "can be defined. One of them has to be None.")

        self.param_change = param_change
        self.param_change_is_abs = param_change_is_abs
        self.prop_change_is_abs = prop_change_is_abs
        self.properties = properties
        self.function = function
        self.name = name
        self.number = None


class StateTransition():

    def __init__(self, start_state=None, end_state=None, parameter=None,
                 transfer_property=None, properties_set_to_zero=None,
                 saved_properties=None, retrieved_properties=None,
                 time_dependency = None, name=""):
        """

        Args:
            start_state (State object): Name of state at which the transition
                starts, If None, then the end_state is created (from state 0)
            end_state (State object): Name of state at which the transition ends
                If None, then the start_date is destroyed (to state 0)
            parameter (Parameter object): Parameter that defines the rate of
                the transition.
            transfer_property (Iterable): 1D Iterable (list, tuple) of two
                ObjectProperty objects. The first object is the property from
                which the values will be transfered and the second object is
                the property to which property the values will be transfered.
            saved_properties (Iterable): 1D Iterable (list, tuple) of
                State Objects for which values should be saved upon transition.
            retrieved_properties (Iterable): 1D Iterable (list, tuple) of
                State Objects for which values should be saved upon transition.
            time_dependency (func): Function that takes a torch tensor
                containing the current timepoints as input and converts it to a
                tensor of factors (using only pytorch functions)
                to then be multiplied with the rates. It is recommended to have
                the time dependency function range from 0 to 1, which makes the
                supplied rates maximum rates.
            name (str): Name of transition, used for data export and readability


        """
        self.start_state = start_state
        self.end_state = end_state
        self.parameter = parameter
        self.saved_properties = saved_properties
        self.retrieved_properties = retrieved_properties
        self.time_dependency = time_dependency
        self.transfer_property = transfer_property
        self.properties_set_to_zero = properties_set_to_zero
        self.name = name
        self.resources = None

        # initialize variable that will be filled during simulation
        self.simulation_mask = torch.HalfTensor([])
        self.transition_positions = torch.HalfTensor([])

class ChangedStartValue():

    def __init__(self,object_property, new_start_values):
        """

        Args:
            property: ObjectProperty object
            new_start_values (float or PropertyGeometry or list of two
                values/ PropertyGeometry): Value at which new objects will be
                initialized, if list, will be initialized randomly between
                first (min_value) and second (max_value) element. Will replace
                original start_values parameter for this object_creation.
        """
        self.object_property = object_property
        self.new_start_values = new_start_values

class ObjectCreation(StateTransition):
    """
    Special class to create new objects.
    """

    def __init__(self, state, parameter, changed_start_values=None,
                 creation_on_objects=False,
                 time_dependency=None, resources = None, name=""):
        """

        Args:
            state:
            parameter:
            changed_start_values: List or tuple of ChangedStartValue objects
            time_dependency:

            resources (int): How many active transitions are allowed. This is
                only implemented for generating new objects. The generation
                process they originate from will be tracked in the object_state
                array at an additional index that will track the generation
                process as transition number. Additionally, the current number
                of objects generated from a specific transition is saved in the
                nb_objects_all_states array as additional dimension with one
                entry per transition. But only generating transitions with
                defined resources will be tracked.
        """
        super().__init__(end_state=state, parameter=parameter,
                         time_dependency=time_dependency, name=name)
        self.changed_start_values = changed_start_values
        self.creation_on_objects = creation_on_objects
        self.resources = resources