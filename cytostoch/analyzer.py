import pandas as pd
import torch
import numpy as np
import os
import re
import copy
import time
from tqdm import tqdm


class Analyzer():

    def __init__(self, simulation=None, data_folder = None, device="gpu",
                 use_free_gpu=False):
        """
            params:
                use_free_gpu (Boolean): Whether to use a free gpu (no memory
                    occupied). If False, will use gpu with highest amount of
                    free memory (for unused gpus, will be usually the most
                    powerful gpu)
        """
        if (simulation is None) & (data_folder is None):
            return ValueError("Simulation or data_folder needs to be defined "
                              "for the Analyzer. But both were None.")

        self.simulation = simulation
        self.data_folder = data_folder

        if (device.lower() == "gpu") & (torch.cuda.device_count() > 0):
            if use_free_gpu:
                # use GPU that is not used already, in case of multiple GPUs
                for GPU_nb in range(torch.cuda.device_count()):
                    if torch.cuda.memory_reserved(GPU_nb) == 0:
                        break
                self.device = "cuda:"+str(GPU_nb)
            else:
                gpu_memory = []
                for GPU_nb in range(torch.cuda.device_count()):
                    device_props = torch.cuda.get_device_properties(GPU_nb)
                    free_memory = (device_props.total_memory
                                   - torch.cuda.memory_reserved(GPU_nb))
                    gpu_memory.append(free_memory)
                highest_free_memory_gpu = np.argmax(np.array(gpu_memory))
                self.device = "cuda:"+str(int(highest_free_memory_gpu))
            self.tensors = torch.cuda
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
            torch.set_default_device(self.device)
        else:
            self.device = "cpu"
            self.tensors = torch
            torch.set_default_tensor_type(torch.FloatTensor)

        # Per default, data folder will be used from the simulation object
        # but if the analyzer is used not directly after a simulation,
        # supplying a different data folder might make sense and is allowed
        if self.simulation is not None:
            self.data_folder = self.simulation.data_folder

    def start(self, time_resolution, max_time, use_assertion_checks = True):
        """
        Fuse all arrays in data_folder together and then extract for each
        simulation specific timepoints.

        Args:
            time_resolution:
            max_time (float): Maximum time to analyze. Should be the minimum
                time that each simulation has run.

        Returns:

        """

        self.time_resolution = time_resolution
        self.max_time = max_time
        self.use_assertion_checks = use_assertion_checks

        print("Starting renaming files...")

        self._rename_files_for_sorting(self.data_folder)

        print("Starting loading time files...")
        params_removed = False
        if self.simulation is None:
            all_times = self._load_data(self.data_folder,
                                        file_name_keyword="times")

            removed_vals_filename = "removed_param_values.feather"
            # load removed_param_value data
            if ((removed_vals_filename not in os.listdir(self.data_folder))
                    & params_removed):
                raise ValueError(
                    "No file for removed parameter values was found. "
                    "But the shape of arrays changes - therefore,"
                    "param values were removed. Maybe the file name"
                    "is not 'removed_param_values.feather' anymore or "
                    "the file was moved to another folder?")
            removed_param_vals_path = os.path.join(self.data_folder,
                                                   removed_vals_filename)
            removed_param_vals = None
            if os.path.exists(removed_param_vals_path):
                removed_param_vals = pd.read_feather(removed_param_vals_path)

            self.array_resizing_dict = {}

            # Reconstruct full time arrays by going through removed param values
            # grouped by timepoint
            # create list of ALL indices for all dimensions from first timepoint
            self.all_dim_list = [torch.IntTensor(range(shape)).to(self.device)
                                 for shape in all_times[0].shape[2:]]

            if removed_param_vals is not None:
                removed_param_vals.groupby(["iteration_nb"]
                                           ).apply(
                    self._build_array_resize_dict)

                print("Starting concatenating arrays with removed params")
                # for time in all_times:

                all_times = self._concat_arrays_with_removed_param_vals(
                    all_times)
            else:
                all_times = torch.concat(all_times)
        else:
            all_times = self.simulation.times.to(self.device)

        print("Finished loading time files.")
        parameters = self._load_parameters()
        time_array = all_times
        # go through each folder that contains sub data
        # There is one folder for each function in DataExtraction used.
        # Time and parameter data is in the parent folder and therefore does
        # not need to be looked at per folder (therefore before this for loop)
        if self.simulation is None:
            all_data_dict = {}
            for folder_name in os.listdir(self.data_folder):
                if not folder_name.startswith("_data"):
                    continue
                data_path = os.path.join(self.data_folder, folder_name)
                if not os.path.isdir(data_path):
                    continue
                all_data_dict[data_path] = {}
                print("Rename files for sorting...")
                self._rename_files_for_sorting(data_path)
                data_keywords = self._get_data_keywords(data_path)
                if "times" in data_keywords:
                    data_keywords.remove("times")
                for data_keyword in data_keywords:
                    new_data = self._load_data(data_path,
                                               file_name_keyword=data_keyword)
                    new_data = self._equalize_object_nb(new_data)
                    try:
                        new_data = torch.concat(new_data)
                    except:
                        concat_func = self._concat_arrays_with_removed_param_vals
                        new_data = concat_func(new_data)
                    all_data_dict[data_path][data_keyword] = new_data
        else:
            all_data_dict = {}
            for data_name, data in self.simulation.all_data.items():
                data_path = os.path.join(self.data_folder, "_data_"+data_name)
                if not os.path.exists(data_path):
                    os.mkdir(data_path)
                all_data_dict[data_path] = {}
                for keyword, data_array in data.items():
                    all_data_dict[data_path][keyword] = data_array.cpu()

        print("Analyzer saving data...")
        for data_path, data_dict in tqdm(all_data_dict.items()):
            all_data = {}
            for data_keyword, new_data in data_dict.items():
                # sort data by number of datapoints for each timepoint
                # since for each different number of datapoints, a new dataframe
                if new_data.shape[1] not in all_data:
                    all_data[new_data.shape[1]] = {}
                all_data[new_data.shape[1]][data_keyword] = new_data
                del new_data
            #print("All data batches loaded.")
            all_data = self._add_single_datapoint_data_to_other_sets(all_data)
            #print("Saving data...")
            self._save_all(all_data, time_array, parameters, data_path)

            del all_data

        del time_array
        del all_times
        del parameters
        torch.cuda.empty_cache()
        # cuda.current_context().memory_manager.deallocations.clear()

        # get expected number of nonzero elements
        # expected_nb_nonzero_elements = ((self.max_timestep+1) *
        #                                 np.prod(all_times.shape[1:]))

        # Load all files of one type from data_folder
        # each file type has a unique starting name, until "_number"

    def _get_data_keywords(self, data_folder):
        data_keyword_finder = re.compile("([\s\S]+)_[\d]+.pt")
        all_data_keyword = set()
        #rename files to be sorted for incrementing time when sorted by name
        max_nb = 0
        for file_name in sorted(os.listdir(data_folder)):
            if file_name.startswith("param_"):
                continue
            data_keyword = data_keyword_finder.search(file_name)
            if data_keyword is None:
                continue
            data_keyword = data_keyword.group(1)
            all_data_keyword.add(data_keyword)
        return all_data_keyword

    def _rename_files_for_sorting(self, data_folder):
        iteration_nb_finder = re.compile("[\D]([\d]+).pt")
        #rename files to be sorted for incrementing time when sorted by name
        max_nb = 0
        for file_name in sorted(os.listdir(data_folder)):
            iteration_nb = iteration_nb_finder.search(file_name)
            if iteration_nb is None:
                continue
            iteration_nb = iteration_nb.group(1)
            max_nb = max(max_nb, int(iteration_nb))

        for file_name in sorted(os.listdir(data_folder)):
            iteration_nb = iteration_nb_finder.search(file_name)
            if iteration_nb is None:
                continue
            iteration_nb = iteration_nb.group(1)
            # check whether renaming is needed
            if len(iteration_nb) == len(str(max_nb)):
                continue
            new_iteration_nb = iteration_nb.zfill(len(str(max_nb)))
            new_file_name = file_name.replace(iteration_nb+".pt",
                                         new_iteration_nb+".pt")
            os.replace(os.path.join(data_folder, file_name),
                      os.path.join(data_folder, new_file_name))
        return None


    def _load_parameters(self):
        parameters = {}
        param_name_finder = re.compile("param_([_\s\S]+).pt")
        # Load all time files, then concatenate together
        for file_name in sorted(os.listdir(self.data_folder)):
            if not file_name.startswith("param_"):
                continue
            param_name = param_name_finder.search(file_name)
            if param_name is None:
                continue
            param_name = param_name.group(1)
            file_path = os.path.join(self.data_folder, file_name)
            parameters[param_name] = torch.load(file_path,
                                                map_location=torch.device(self.device))
        return parameters


    def _load_data(self, data_folder, file_name_keyword):
        keyword_finder = re.compile(file_name_keyword+"_[\d]+.pt")
        all_data = []
        old_shape = ()
        new_data_array = None
        # Load all data files, then concatenate together
        file_nb = 0
        for file_name in sorted(os.listdir(data_folder)):
            if keyword_finder.search(file_name) is None:
                continue
            file_nb += 1
            file_path = os.path.join(data_folder, file_name)
            new_data = torch.load(file_path,
                                   map_location=torch.device(self.device))
            # new_data = new_data.unsqueeze(0)
            new_shape = new_data.shape
            # combine all time and data arrays that have the same shape
            # by opening a new array when the shape changes and concatenating next
            # arrays to it
            if new_shape != old_shape:
                if new_data_array is not None:
                    # compare shapes without the first two dimension (since the
                    # second dimension indicates the number of objects,
                    # which does not inform about whether parameter values
                    # were removed)
                    all_data.append(torch.concat(new_data_array))
                new_data_array = [new_data]
                old_shape = new_shape
            else:
                new_data_array.append(new_data)

        all_data.append(torch.concat(new_data_array))

        return all_data

    def _build_array_resize_dict(self, removed_params):
        iteration_nb = removed_params["iteration_nb"].unique()[0]
        removed_params = removed_params.set_index("dimension")
        for dim in range(len(self.all_dim_list)):
            if (dim + 1) not in removed_params.index.values:
                continue
            removed_params_dim = removed_params.loc[dim + 1]
            if type(removed_params_dim) == pd.Series:
                removed_params_dim = pd.DataFrame(removed_params_dim)
                removed_params_dim = removed_params_dim.transpose()
            removed_positions = removed_params_dim["position"].values
            # set all removed positions to -1, so that removed positions are
            # clear from

            # removed positions are from the perspective of the actual
            # tensor size at that moment - which includes previously removed
            # positions in that dimension
            # therefore, create new tensor from that perspective,
            # with removed positions set to -1
            nb_nonzero = torch.count_nonzero(self.all_dim_list[dim] > -1)
            new_indices = torch.IntTensor(range(0, nb_nonzero))
            new_indices[removed_positions] = -1
            # removed positions will also shift the dimensions
            # to account for that map all non -1 positions
            # to integers from 0 to (len(non -1 positions) - 1)
            nb_nonzero = torch.count_nonzero(new_indices > -1)
            re_numbered_indices = torch.IntTensor(range(0, nb_nonzero))
            new_indices[new_indices > -1] = re_numbered_indices
            new_indices = new_indices.to(self.device)
            self.all_dim_list[dim][self.all_dim_list[dim] > -1] = new_indices
        # save array resizing information for all dimensions in
        # current iteration
        self.array_resizing_dict[iteration_nb] = copy.deepcopy(self.all_dim_list)

    def _concat_arrays_with_removed_param_vals(self, data_list):
        # use this index list to create equal sized arrays
        current_iteration_nb = 0
        array_resizer = None
        data_concat = torch.FloatTensor().to(self.device)
        # go through all time arrays while keeping track of the current
        # iteration_nb
        # change their size
        new_data_list = []
        for data in data_list:
            # get the current array resizer
            array_resizer = self.array_resizing_dict.get(current_iteration_nb,
                                                         array_resizer)
            current_iteration_nb += data.shape[0]
            # if no array resizer has been used so far, concatenate tensors
            if array_resizer is None:
                data_concat = torch.concat([data_concat, data])
                continue
            # once an array resizer has been available (once the first iteration
            # with an array resizer is reached) - which is the case once
            # the first parameters were removed in the array
            new_data = data
            for dim, dim_resizer in enumerate(array_resizer):
                ended_sim_marker = torch.clone(dim_resizer)

                # index_select does not work with index -1
                dim_resizer[dim_resizer < 0] = 0
                # resize the array by selecting the dimensions based on
                # the array resizer, also duplicating positions,
                # in order to get the array to the full size (same size as
                # first timepoint)
                # do this for each position separately (since index_select
                # only allows one dimension at a time)
                new_data = torch.index_select(new_data, dim=dim+2,
                                               index=dim_resizer)

                # mark all positions with ended simulations
                # as above max time (max_time + 2* time_resolution)
                # by first multiplying by 0 and
                # then adding max_time + 2* time_resolution
                ended_sim_marker[ended_sim_marker >= 0] = 1
                ended_sim_marker[ended_sim_marker < 0] = 0
                shape = [1] * len(array_resizer)
                shape[dim] = -1
                ended_sim_marker = ended_sim_marker.view(shape)
                ended_sim_marker = ended_sim_marker.expand(new_data.shape)
                ended_sim_marker = torch.clone(ended_sim_marker)
                new_data *= ended_sim_marker
                # To mark positions with ended simulations,
                # also subtract one in removed positions,
                # to also transform 0 values to below 0
                ended_sim_marker[ended_sim_marker == 0] = (self.max_time +
                                                           self.time_resolution
                                                           * 2)
                ended_sim_marker[ended_sim_marker > 0] = 0
                new_data += ended_sim_marker
            new_data[new_data < 0] = -1
            # concatenate newly enlarged array to previous arrays
            new_data_list.append(new_data)

        data_concat = torch.concat([data_concat, *new_data_list])

        return data_concat

    def _get_timestep_changes_and_repeats(self, all_times):

        # For each simulation, the first timepoint in a timestep should be used
        # the remaining timepoints discarded
        # Calculate difference of array with array shifted by 1 in time
        # dimension. Thereby only the first timepoint for each timestep is 1.
        timestep_changes = all_times[1:] - all_times[:-1]

        # due to shifted difference time jumps are also
        # shifted by one position (first element in difference should be for
        # the second element in whole list)
        # the first element should be used, therefore a tensor with value 1
        # will be added to the beginning
        expanded_shape = [1, *timestep_changes.shape[1:]]
        expanded_one_tensor = torch.ByteTensor([1]).expand(expanded_shape)
        timestep_changes = torch.cat([expanded_one_tensor,
                                      timestep_changes], dim=0)

        # shift timestep_changes to get the last timepoint for each simulation
        # Use last timepoint at end of array
        # this timepoint can either be 0 (after max timeframe)
        # or it can be >0, in which case it must be the max_timestep
        # for both cases, the last timepoint can be used as is
        shape = timestep_changes.shape
        timestep_changes_end = torch.cat([timestep_changes[1:],
                                          -all_times[-1].expand((1,
                                                                 *shape[1:]))],
                                         dim=0)

        # all timepoint changes above 0 indicate that the timepoint before
        # was lower, which means that the timepoint was not the last one
        # therefore set the jumps tot he max number of steps,
        # so that they will be 0 after adding the max_timesteps again
        timestep_changes_end[timestep_changes_end >= 0] = - self.max_timestep
        # add the max timesteps to get the number of timesteps missing at the
        # end
        timestep_changes_end += self.max_timestep

        # remove negative timejumps (from max timestep to timestep afterwards)
        timestep_changes[timestep_changes < 0] = 0
        # add time jumps at the end to current timejumps
        timestep_changes[timestep_changes_end > 0] += timestep_changes_end[
            timestep_changes_end > 0]

        # it might also be that some simulations do not have a single time step
        # within the time range (except time=0)
        # to prevent the code to throw an error since array sizes cannot  be
        # homogenized, correct for that
        jump_sum = torch.sum(timestep_changes, dim=0)
        jump_sum_mask = jump_sum.unsqueeze(0).expand(timestep_changes_end.shape[0],
                                                     *jump_sum.shape)
        timestep_changes_zero = torch.zeros(timestep_changes_end.shape)
        timestep_changes_zero[(jump_sum_mask == 1) &
                              (timestep_changes > 0)] = self.max_timestep
        timestep_changes[timestep_changes_zero > 0] += timestep_changes_zero[
            timestep_changes_zero > 0]

        # make sure that the total number of jumps is the same for each
        # simulation
        # and that this number is the expected (max) number of timesteps
        jump_sum = torch.sum(timestep_changes, dim=0)
        assert ((jump_sum.min() == jump_sum.max()) &
                (jump_sum.min() == self.max_timestep + 1))

        return timestep_changes

    def _get_equal_timesteps_array(self, input_array, timestep_changes,
                                   repeated_time_mask=None,
                                   type=None):
        # get timestep changes to same shape a input array...
        # firt change view though

        input_array = input_array.expand([timestep_changes.shape[0],
                                          input_array.shape[1],
                                          *timestep_changes.shape[2:]])

        timestep_changes = timestep_changes.expand(input_array.shape)

        # move time axis to last position so that all timepoints of the
        # same simulation are directly after one another
        # thereby, repeating timepoints to fill up missing ones
        # does not need to an unordered array
        timestep_changes = torch.moveaxis(timestep_changes, 0, -1)

        timestep_repeats = timestep_changes[timestep_changes > 0]
        timestep_repeats = timestep_repeats.to(torch.int)

        # extract data from all data arrays
        data_timesteps_flat = torch.moveaxis(input_array,0,-1)[timestep_changes
                                                               > 0]

        # add missing timepoints that the algorithm jumped over (since time
        # resolution was smaller than time difference in algorithm)
        # do that for all data arrays
        data_timesteps_flat = torch.repeat_interleave(data_timesteps_flat,
                                                      repeats=timestep_repeats)

        # now reshape for all data arrays
        data_timesteps = torch.reshape(data_timesteps_flat,
                                       (*timestep_changes.shape[:-1],-1))

        # and move time axis back to dimension 0
        data_timesteps = torch.moveaxis(data_timesteps, -1, 0)
        if self.use_assertion_checks & (type == "time"):
            # make sure that the maximum timestep is the expected max_timestep
            assert data_timesteps_flat.max() == self.max_timestep

            # used_timepoints = torch.repeat_interleave(torch.moveaxis(times_tmp,
            #                                                          0,-1)
            #                                           [timestep_changes > 0],
            #                                           repeats=
            #                                           timestep_repeats)
            # # make sure that the maximum timepoint is not larger than the allower
            # # maximum timepoint
            # assert used_timepoints.max() <= (self.max_time +
            #                                  self.time_resolution)

        if repeated_time_mask is None:
            repeated_time_mask = self._get_repeated_timesteps_mask(data_timesteps)
        # Use mask to set values of all data arrays to nan
        data_timesteps = data_timesteps.to(torch.float32)
        data_timesteps[repeated_time_mask.expand(data_timesteps.shape)] = float("nan")

        return data_timesteps, repeated_time_mask

    def _get_repeated_timesteps_mask(self, time_array):
        # create array for all timepoints
        timepoints = torch.linspace(0,self.max_timestep,
                                    int(self.max_timestep)+1)
        timepoints = timepoints.view([-1] + [1]*len(time_array.shape[1:]))
        timepoints = timepoints.expand(time_array.shape)

        # subtract timepoint array from new_times to get positions that should
        # be used. Only positions with value == 0 should be used, set other
        # values to nan.
        repeated_time_mask = (time_array - timepoints) != 0

        return repeated_time_mask


    def _equalize_object_nb(self, data):
        # equalize second dimension (number of objects), if it changed
        nb_objects = np.array([new_data.shape[1]
                               for new_data in data])
        max_nb_objects = nb_objects.max()
        for nb, new_data in enumerate(data):
            current_nb_objects = new_data.shape[1]
            nb_objects_to_add = max_nb_objects - current_nb_objects
            if nb_objects_to_add == 0:
                continue
            objects_to_add = torch.zeros((new_data.shape[0],
                                          nb_objects_to_add,
                                          *new_data.shape[2:]))
            data[nb] = torch.cat([new_data, objects_to_add],
                                         dim=1)
        return data

    def _add_single_datapoint_data_to_other_sets(self, all_data):
        # if there is just one datapoint for each timepoint, then data can
        # be added to other data in the same dataframe again
        # (can be expanded to same size)
        if 1 not in all_data:
            return all_data
        data_shapes = list(all_data.keys())
        data_shapes.remove(1)
        keywords = all_data[1].keys()
        for data_shape in data_shapes:
            for keyword in keywords:
                all_data[data_shape][keyword] = all_data[1][keyword]
        # if data was copied to at least one other data shape, delete it
        if len(data_shapes) > 0:
            del all_data[1]
        return all_data

    def _save_all(self, all_data, time_array, parameters, data_folder):
        data_shapes = None
        for nb_datapoints, data_dict in all_data.items():
            # get datashapes
            all_data_shapes = [data.shape for data in data_dict.values()]

            if (nb_datapoints != 1):
                data_shapes = [shape for shape in all_data_shapes
                               if shape[2] != 1]
            if data_shapes is None:
                data_shapes = [shape for shape in all_data_shapes]
            if len(data_shapes) == 0:
                data_shapes = [shape for shape in all_data_shapes]

            data_shape = data_shapes[0]

            # prepare data to be saved in dataframe
            # for that needs to be of shape (datapoints, columns)
            data_values = []
            column_names = []
            # first add time data
            # print(time_array.shape, data_shape)
            # print("new data shape: ", data_shape)
            # for one_data_shape in data_shapes:
            #     print(one_data_shape)

            # if data_shape at last two positions (simulations and parameters)
            # is 1 then the data is simulation invariant
            if (data_shape[-2] == 1) & (data_shape[-1] == 1):
                time_shape = [slice(None) for _ in time_array.shape]
                time_shape[-1] = 0
                time_shape[-2] = 0
                time_array = time_array[time_shape]

            # get equal dimensions for time array and data shape
            while len(time_array.shape) < len(data_shape):
                time_array = time_array.unsqueeze(-1)

            data_values.append(time_array.cpu().expand(data_shape).flatten().unsqueeze(1))
            column_names.append("time")
            # add information about simulation number
            simulation_array_shape = [1 for _ in data_shape]
            simulation_array_shape[2] = data_shape[2]
            simulation_array = torch.arange(data_shape[2]).view(
                simulation_array_shape)
            data_values.append(simulation_array.cpu().expand(
                data_shape).flatten().unsqueeze(1))
            column_names.append("simulation_nb")

            # next add all simulation parameters
            for name, parameter in parameters.items():
                # print(parameter.shape, data_shape)
                # print(torch.Tensor(parameter).unsqueeze(2).unsqueeze(1).expand(data_shape).shape)
                # print(torch.Tensor(parameter).unsqueeze(1).unsqueeze(2).shape)

                # ONLY TAKE PARAMETER VALUE BEFORE TIMESWITCH!
                if len(parameter.shape) > 2:
                    parameter = parameter[:, :, 0]

                # if recording of values did not start at t = 0,
                # parameter size might be too large in first dimension
                # since it contains the parameter value for each timepoint at
                # the defined resolution, even before recording of data started
                # therefore select the last data of parameter that corresponds
                # to the parameter values after data recording started
                if parameter.shape[0] > data_shape[0]:
                    parameter = parameter[-data_shape[0]:]

                # if the last dimensions is larger for parameter
                # get the same size of the last dimension as for the data_shape
                if parameter.shape[-1] > data_shape[-1]:
                    param_idx = [slice(shape) for shape in parameter.shape]
                    param_idx[-1] = slice(data_shape[-1])
                    parameter = parameter[*param_idx]

                data_values.append(torch.Tensor(parameter).unsqueeze(1).unsqueeze(2).cpu().expand(
                    data_shape).flatten().unsqueeze(1))
                column_names.append(name)

            # now add all data
            for column_name, data in data_dict.items():
                data_values.append(data.cpu().expand(data_shape)
                                   .flatten().unsqueeze(1))
                column_names.append(column_name)

            data = torch.concat(data_values, dim=1)

            dataframe = pd.DataFrame(data=data, columns=column_names)
            # file_name = "_".join(column_names[1:])
            file_name = "data"
            #print("Saving dataframe...")
            dataframe.to_feather(os.path.join(data_folder,
                                              file_name + ".feather"))