import torch
import numpy as np
import os
import re
import time

class Analyzer():

    def __init__(self, simulation=None, data_folder = None):
        """

        """
        if (simulation is None) & (data_folder is None):
            return ValueError("Simulation or data_folder needs to be defined "
                              "for the Analyzer. But both were None.")

        self.simulation = simulation

        # Per default, data folder will be used from the simulation object
        # but if the analyzer is used not directly after a simulation,
        # supplying a different data folder might make sense and is allowed
        if data_folder is None:
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

        iteration_nb_finder = re.compile("[\D]([\d]+).pt")
        all_times = torch.FloatTensor([])
        #rename files to be sorted for incrementing time when sorted by name
        max_nb = 0
        for file_name in sorted(os.listdir(self.data_folder)):
            iteration_nb = iteration_nb_finder.search(file_name).group(1)
            max_nb = max(max_nb, int(iteration_nb))

        for file_name in sorted(os.listdir(self.data_folder)):
            iteration_nb = iteration_nb_finder.search(file_name).group(1)
            new_iteration_nb = iteration_nb.zfill(len(str(max_nb)))
            new_file_name = file_name.replace(iteration_nb+".pt",
                                         new_iteration_nb+".pt")
            os.rename(os.path.join(self.data_folder, file_name),
                      os.path.join(self.data_folder, new_file_name))

        # Load all time files, then concatenate together
        for file_name in sorted(os.listdir(self.data_folder)):
            if not file_name.startswith("time"):
                continue
            file_path = os.path.join(self.data_folder, file_name)
            new_times = torch.load(file_path)
            new_times = new_times.unsqueeze(0)
            all_times = torch.cat((all_times, new_times), dim=0)

        print("All files loaded.")
        times_tmp = torch.clone(all_times)

        # Simulations are asynchronous.
        # Therefore discard timepoints above the maximum time, to have the same
        # number of timesteps for all simulations
        all_times[all_times >= (max_time+time_resolution)] = 0

        # Floor division by time resolution to get time steps
        all_times = torch.div(all_times, time_resolution, rounding_mode="floor")

        # For each simulation, the first timepoint in a timestep should be used
        # the remaining timepoints discarded
        # Calculate difference of array with array shifted by 1 in time
        # dimension. Thereby only the first timepoint for each timestep is 1.
        time_jumps = all_times[1:] - all_times[:-1]

        # due to shifted difference time jumps are also
        # shifted by one position (first element in difference should be for
        # the second element in whole list)
        # the first element should be used, therefore a tensor with value 1
        # will be added to the beginning
        expanded_shape = [1,*time_jumps.shape[1:]]
        expanded_one_tensor = torch.ByteTensor([1]).expand(expanded_shape)
        time_jumps = torch.cat([expanded_one_tensor, time_jumps], dim=0)

        # Get how many timepoints at the end of each simulation were jumped
        # over, use the expected max timestep for that
        max_timestep = max_time/time_resolution
        # shift time_jumps to get the last timepoint for each simulation
        # Use last timepoint at end of array
        # this timepoint can either be 0 (after max timeframe)
        # or it can be >0, in which case it must be the max_timestep
        # for both cases, the last timepoint can be used as is
        time_jumps_end = torch.cat([time_jumps[1:],
                                    -all_times[-1].expand((1,*time_jumps.shape[1:]))],
                                   dim=0)

        # all timepoint changes above 0 indicate that the timepoint before
        # was lower, which means that the timepoint was not the last one
        # therefore set the jumps tot he max number of steps,
        # so that they will be 0 after adding the max_timesteps again
        time_jumps_end[time_jumps_end >= 0] = - max_timestep
        # add the max timesteps to get the number of timesteps missing at the
        # end
        time_jumps_end += max_timestep

        # remove negative timejumps (from max timestep to timestep afterwards)
        time_jumps[time_jumps < 0] = 0
        # add time jumps at the end to current timejumps
        time_jumps[time_jumps_end > 0] += time_jumps_end[time_jumps_end > 0]

        # make sure that the total number of jumps is the same for each
        # simulation
        # and that this number is the expected (max) number of timesteps
        jump_sum = torch.sum(time_jumps, dim=0)
        assert ((jump_sum.min() == jump_sum.max()) &
                (jump_sum.min() == max_timestep + 1))

        # move time axis to last position so that all timepoints of the
        # same simulation are directly after one another
        # thereby, repeating timepoints to fill up missing ones
        # does not need to an unordered array
        time_jumps = torch.moveaxis(time_jumps, 0, -1)

        time_jumps_changes = time_jumps[time_jumps > 0]
        time_jumps_changes = time_jumps_changes.to(torch.int)

        # extract data from all data arrays
        new_times = torch.moveaxis(all_times,0,-1)[time_jumps > 0]

        # add missing timepoints that the algorithm jumped over (since time
        # resolution was smaller than time difference in algorithm)
        # do that for all data arrays
        new_times = torch.repeat_interleave(new_times,
                                            repeats=time_jumps_changes)


        # make sure that the maximum timestep is the expected max_timestep
        assert new_times.max() == max_timestep

        if use_assertion_checks:
            used_timepoints = torch.repeat_interleave(torch.moveaxis(times_tmp,
                                                                     0,-1)
                                                      [time_jumps > 0],
                                                      repeats=
                                                      time_jumps_changes)
            # make sure that the maximum timepoint is not larger than the allower
            # maximum timepoint
            assert used_timepoints.max() <= (max_time + time_resolution)

        # now reshape for all data arrays
        time_array = torch.reshape(new_times,(*time_jumps.shape[:-1],-1))
        # and move time axis back to dimension 0
        time_array = torch.moveaxis(time_array, -1, 0)

        # create array for all timepoints
        timepoints = torch.linspace(0,max_timestep, int(max_timestep)+1)
        timepoints = timepoints.view([-1] + [1]*len(time_array.shape[1:]))
        timepoints = timepoints.expand(time_array.shape)

        # subtract timepoint array from new_times to get positions that should
        # be used. Only positions with value == 0 should be used, set other
        # values to nan.
        repeated_time_mask = (time_array - timepoints) != 0

        # Use mask to set values of all data arrays to nan
        time_array[repeated_time_mask] = float("nan")

        # get expected number of nonzero elements
        expected_nb_nonzero_elements = ((max_timestep+1) *
                                        np.prod(all_times.shape[1:]))
        print(expected_nb_nonzero_elements)

        print(new_times.shape[0])
        print(new_times.shape[0] - expected_nb_nonzero_elements)

        # Load all files of one type from data_folder
        # each file type has a unique starting name, until "_number"
