import torch
import numpy as np
import time
import math
import numba
import functools


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
        position_array[~mask] = float("nan")
        position_array = torch.gather(position_array, dim=1, index=idx)
        position_array = position_array[:, :max_nb_objects]

        length_array[~mask] = float("nan")
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
        max_values = properties[0].max_value - properties[0].array

        # subtract values of each property in the same dimension from max value
        # to get
        if len(properties) > 1:
            for property in properties[1:]:
                max_values -= property.array

        return max_values

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
                 print_regularly=False, **kwargs):
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

        self.resolution = kwargs.get("resolution", None)

        implemented_operations = {}
        new_operation = self._operation_2D_to_1D_density
        implemented_operations["2D_to_1D_density"] = new_operation
        implemented_operations["raw"] = self._operation_raw
        implemented_operations["global"] = self._operation_global

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

    def extract(self, simulation_object, regular_print=False):

        if self.state_groups is None:
            data, _ = self.operation(self.dimensions, simulation_object,
                                     regular_print=regular_print,
                                **self.kwargs)
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
                                                  **self.kwargs)

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
                                                 **self.kwargs)
                # extract the actual data and not auxiliary information
                # which would be the same between different state groups
                # due to same underlying space structure
                for name, data_vals in data.items():
                    # data_col_names contains all names of the actual data
                    # which is different between state groups
                    if name in data_col_names:
                        all_data[name+"_"+group_name] = data_vals
                    else:
                        # add auxiliary information once
                        if name not in all_data.keys():
                            all_data[name] = data_vals

        return all_data

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
        for state in state_numbers:
            mask = (mask | (object_states == state))

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
                property_array = property.array
                # if regular_print:
                #     property_array = property.array.unsqueeze(0)
                # else:
                #     property_array = torch.concat(property.array_buffer)

            property_array[~mask] = float("nan")
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
            data_dict["mass_"+name] = (mean_values * object_number).cpu()
            all_data_columns.append("mean_"+name)
            all_data_columns.append("mean_inside_"+name)
            all_data_columns.append("mass_"+name)

        return data_dict, all_data_columns

    def _operation_raw(self, dimensions, **kwargs):
        data_dict = {}
        data_dict["position"] = dimensions[0].positions[0].array.cpu()
        data_dict["length"] = dimensions[0].length.array.cpu()
        return data_dict

    def _operation_2D_to_1D_density_numba(self, dimensions, simulation_object,
                                    state_numbers=None, regular_print=False,
                                    resolution=0.2, **kwargs):
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
        position_array = dimensions[0].position.array.clone()
        # if regular_print:
        #     position_array = dimensions[0].position.array
        # else:
        #     position_array = torch.concat(dimensions[0].position.array_buffer)

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
            position_array[~mask] = float("nan")
            position_array = torch.gather(position_array, dim=1, index=idx)
            position_array = position_array[:, :max_nb_objects]

            length_array[~mask] = float("nan")
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

        positions = torch.arange(min_position, max_position+resolution*0.9,
                                 resolution).to(device)

        # print(position_array.shape, positions.shape)
        all_data = torch.zeros((position_array.shape[0], positions.shape[0],
                                *position_array.shape[2:])).to(device)

        # only if at least one element is True, analyze the data
        if mask.sum() > 0:

            position_dimension = int(round((max_position - min_position)
                                           / resolution,5))
            # print(1, time.time() - start)
            start = time.time()
            # create tensors on correct device
            tensors = simulation_object.tensors

            # data type depends on dimension 0 - since that is the number of
            # different int values needed (int8 for <=256; int16 for >=256)
            # (dimension 0 is determined by max_x of neurite / resolution)
            if (position_dimension+1) < 256:
                indices_tensor = torch.ByteTensor
                indices_dtype = torch.uint8
            else:
                indices_tensor = torch.ShortTensor
                indices_dtype = torch.short

            # extract positions of the array that actually contains objects
            # crop data so that positions that don't contain objects
            # are excluded
            #objects_array = ~torch.isnan(position_array)
            #positions_object = torch.nonzero(objects_array)
            #min_pos_with_object = positions_object[:,0].min()

            position_start = position_array
            #max_nb_objects = position_start.shape[0]
            #position_start = position_start[min_pos_with_object:max_nb_objects]

            # transform object properties into multiples of resolution
            # then transform length into end position
            position_start = torch.div(position_start, resolution,
                                       rounding_mode="floor")#.to(torch.short)
            position_start = torch.unsqueeze(position_start, 1)

            position_end = length_array.unsqueeze(1)

            #position_end = position_end[min_pos_with_object:max_nb_objects]
            position_end = torch.div(position_end + position_array.unsqueeze(1),
                                     resolution,
                                     rounding_mode="floor")#.to(indices_dtype)

            # # remove negative numbers to only look at inside the neurite
            # position_start[position_start < 0] = 0
            # position_start = position_start#.to(indices_dtype)

            # create indices array which each number
            # corresponding to one position in space (in dimension 0)
            indices = np.linspace(0,position_dimension, position_dimension+1)
            indices = np.expand_dims(indices,
                                     tuple(range(1,len(position_start.shape)-1)))
            indices = indices_tensor(indices).unsqueeze(0).to(device=device)

            # split by simulations to reduce memory usage, if needed
            # otherwise high memory usage leads to
            # massively increased processing time
            # with this split, processing time increases linearly with
            # array size (up to a certain max array size beyond which
            # the for loop leads to a supralinear increase in processing time)

            nb_objects = position_start.shape[2]
            # find way to dynamically determine ideal step size!
            step_size = nb_objects#5
            nb_steps = int(nb_objects/step_size)
            start_nb_objects = torch.linspace(0, nb_objects-step_size, nb_steps)

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
                            >= position_start[:,:, start_nb_object:end_nb_object]) &
                            (indices.expand(nb_timepoints,-1, step_size,
                                            *position_start.shape[3:])
                            <= position_end[:,:, start_nb_object:end_nb_object]))

                # then sum across microtubules to get number of MTs at each position
                data_array_sum = torch.sum(data_array, dim=2, dtype=torch.int16)

                all_data = all_data + data_array_sum
                # val_tmp = all_data[:2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

        # print(3, time.time() - start)
        start = time.time()
        data_array = all_data[:,:-1]

        dimensions = [-1] + [1] * (len(data_array.shape) - 2)
        positions = positions.view(*dimensions)[:-1]
        positions = positions.unsqueeze(0).expand([data_array.shape[0],
                                                   *positions.shape])

        data_dict = {}
        data_dict["1D_density_position"] = positions.cpu()
        data_dict["1D_density"] = data_array.cpu()

        del data_array
        del positions
        torch.cuda.empty_cache()

        return data_dict, ["1D_density"]

    def _operation_2D_to_1D_density(self, dimensions, simulation_object,
                                    state_numbers=None, regular_print=False,
                                    resolution=0.2, **kwargs):
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
            position_array[~mask] = float("nan")
            position_array = torch.gather(position_array, dim=1, index=idx)
            position_array = position_array[:, :max_nb_objects]

            length_array[~mask] = float("nan")
            length_array = torch.gather(length_array, dim=1, index=idx)
            length_array = length_array[:, :max_nb_objects]

        # create boolean data array later by expanding each microtubule in space
        # size of array will be:
        # (max_position of neurite / resolution) x nb of microtubules
        min_position = dimensions[0].positions[0].min_value
        if min_position is None:
            min_position = 0
        max_position = dimensions[0].positions[0].max_value

        device = simulation_object.device

        positions = torch.arange(min_position, max_position+resolution*0.9,
                                 resolution).to(device)

        # print(position_array.shape, positions.shape)
        all_data = torch.zeros((position_array.shape[0], positions.shape[0],
                                *position_array.shape[2:])).to(device)

        # only if at least one element is True, analyze the data
        if mask.sum() > 0:

            position_dimension = int(round((max_position - min_position)
                                           / resolution,5))
            # print(1, time.time() - start)
            start = time.time()
            # create tensors on correct device
            tensors = simulation_object.tensors

            # data type depends on dimension 0 - since that is the number of
            # different int values needed (int8 for <=256; int16 for >=256)
            # (dimension 0 is determined by max_x of neurite / resolution)
            if (position_dimension+1) < 256:
                indices_tensor = torch.ByteTensor
                indices_dtype = torch.uint8
            else:
                indices_tensor = torch.ShortTensor
                indices_dtype = torch.short

            # extract positions of the array that actually contains objects
            # crop data so that positions that don't contain objects
            # are excluded
            #objects_array = ~torch.isnan(position_array)
            #positions_object = torch.nonzero(objects_array)
            #min_pos_with_object = positions_object[:,0].min()

            position_start = position_array
            #max_nb_objects = position_start.shape[0]
            #position_start = position_start[min_pos_with_object:max_nb_objects]

            # transform object properties into multiples of resolution
            # then transform length into end position
            position_start = torch.div(position_start, resolution,
                                       rounding_mode="floor")#.to(torch.short)
            position_start = torch.unsqueeze(position_start, 1)

            position_end = length_array.unsqueeze(1)

            #position_end = position_end[min_pos_with_object:max_nb_objects]
            position_end = torch.div(position_end + position_array.unsqueeze(1),
                                     resolution,
                                     rounding_mode="floor")#.to(indices_dtype)

            # # remove negative numbers to only look at inside the neurite
            # position_start[position_start < 0] = 0
            # position_start = position_start#.to(indices_dtype)

            # create indices array which each number
            # corresponding to one position in space (in dimension 0)
            indices = np.linspace(0,position_dimension, position_dimension+1)
            indices = np.expand_dims(indices,
                                     tuple(range(1,len(position_start.shape)-1)))
            indices = indices_tensor(indices).unsqueeze(0).to(device=device)

            # split by simulations to reduce memory usage, if needed
            # otherwise high memory usage leads to
            # massively increased processing time
            # with this split, processing time increases linearly with
            # array size (up to a certain max array size beyond which
            # the for loop leads to a supralinear increase in processing time)

            nb_objects = position_start.shape[2]

            # find way to dynamically determine ideal step size!
            step_size = nb_objects

            # calculate total memory needed
            element_size = indices.element_size()
            nb_elements = (indices.numel() * position_start.shape[0] *
                           np.product(position_start.shape[3:]))
            # divide by 1024 to prevent overflow
            expected_size = 2 * ((nb_elements*element_size)/1024*step_size)
            free_gpu_memory = (torch.cuda.get_device_properties(0).total_memory
                               - torch.cuda.memory_reserved(0))/1024
            if expected_size > free_gpu_memory:
                step_size = math.floor(nb_objects /
                                       math.ceil(expected_size/free_gpu_memory))
            else:
                step_size = nb_objects

            nb_steps = math.ceil(nb_objects/step_size)
            start_nb_objects = torch.linspace(0, nb_objects-step_size, nb_steps)
            # print(2, time.time() - start)
            start = time.time()
            nb_timepoints = position_start.shape[0]
            for start_nb_object in start_nb_objects:
                end_nb_object = int((start_nb_object + step_size).item())
                start_nb_object = int(start_nb_object.item())
                # create boolean data array later by expanding each microtubule in space
                # use index array to set all positions in boolean data array to True
                # that are between start point and end point
                data_array = ((indices.expand(nb_timepoints, -1, step_size,
                                              *position_start.shape[3:])
                            >= position_start[:,:, start_nb_object:end_nb_object]) &
                            (indices.expand(nb_timepoints, -1, step_size,
                                              *position_start.shape[3:])
                            <= position_end[:,:, start_nb_object:end_nb_object]))

                # remove data from objects that have no length
                # (e.g. objects that should only have one property will have 0
                #  in the other property and therefore should have a density
                #  of 0)
                data_array[:,:,length_array[0,
                               start_nb_object:end_nb_object] == 0] = 0
                # then sum across microtubules to get number of MTs at each position
                data_array_sum = torch.sum(data_array, dim=2, dtype=torch.int16)
                all_data = all_data + data_array_sum
                # val_tmp = all_data[:2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                del data_array
                del data_array_sum
                torch.cuda.empty_cache()

        # print(3, time.time() - start)
        start = time.time()
        data_array = all_data[:,:-1]

        dimensions = [-1] + [1] * (len(data_array.shape) - 2)
        positions = positions.view(*dimensions)[:-1]
        positions = positions.unsqueeze(0).expand([data_array.shape[0],
                                                   *positions.shape])

        data_dict = {}
        data_dict["1D_density_position"] = positions.cpu()
        data_dict["1D_density"] = data_array.cpu()
        del positions

        torch.cuda.empty_cache()

        return data_dict, ["1D_density"]

class ObjectProperty():

    def __init__(self, min_value=None, max_value=None, start_value=[0,1],
                 initial_condition=None, name=""):
        """
        Args:
            min_value (float or PropertyGeometry): lower limit on property
                value, objects can't go below this value (flooring)
            max_value (float or PropertyGeometry): upper limit on property
                value, objects can't go above this value (ceiling)
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

    def __init__(self, values, type="rates", convert_half_lifes=True,
                 name=""):
        """

        Args:
            values (Iterable): 1D Iterable (list, numpy array) of all values
                which should be used. If type is half-lifes, then values will
                be converted to rates if convert_half_lifes is True.
            type (String): Type of values supplied, can be "rates", "half-lifes"
                or "other". For half-lifes, values will be converted to rates
                if convert_half_lifes is True.
            name (string): Name of parameter
        """
        if (type == "half-lifes") & (convert_half_lifes):
            lifetime_to_rates_factor = np.log(np.array([2]))
            self.values = torch.HalfTensor(lifetime_to_rates_factor / values)
        else:
            self.values = torch.HalfTensor(values)

        self.name = name
        self.number = None
        self.value_array = torch.HalfTensor([])



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
                 time_dependency=None, name=""):
        """

        Args:
            state:
            parameter:
            changed_start_values: List or tuple of ChangedStartValue objects
            time_dependency:
        """
        super().__init__(end_state=state, parameter=parameter,
                         time_dependency=time_dependency, name=name)
        self.changed_start_values = changed_start_values
        self.creation_on_objects = creation_on_objects
