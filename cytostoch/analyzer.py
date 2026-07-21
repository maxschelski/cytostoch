import pandas as pd
import torch
import numpy as np
import os
import re
import copy
import time
from tqdm import tqdm
import numba
import cytostoch.simulation
from cytostoch import simulation
import math
import shutil
import seaborn as sb
import scipy
import functools

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
def _object_length_in_range(start_length_in_range, end_length_in_range,
                            range_start, range_end,
                            start_nb_object, end_nb_object,
                            position_array, length_array, orientation,
                            nb_parallel_cores):

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
        while object_pos < last_object_pos:
            if ((not math.isnan(length_array[object_pos, sim_id, param_id]))):
                start = position_array[object_pos, sim_id, param_id]
                if orientation is not None:
                    # if the orientation is opposite, length reduces position
                    end = (start + orientation[object_pos, sim_id, param_id] *
                           length_array[object_pos, sim_id, param_id])

                    # if the orientation is opposite, property vals reduced
                    # the end value and therefore end is smaller than
                    # start, therefore switch end and start so that start is
                    # smaller than end again for the density calculation below
                    if orientation[object_pos, sim_id, param_id] == -1:
                        end_tmp = end
                        end = start
                        start = end_tmp
                else:
                    end = (start + length_array[object_pos, sim_id, param_id])

                if (start < range_end) & (end > range_start):
                    # if the object starts after the range start,
                    # the start length in range is 0, since the beginning is
                    # already in the range
                    if start >= range_start:
                        start_length_in_range[object_pos, sim_id, param_id] = 0
                    else:
                        # otherwise, the start length in range is the difference
                        # between the start pos in range and the start of the
                        # object
                        start_length_in_range[object_pos,
                                           sim_id, param_id] = (range_start -
                                                                start)

                    # if the object end is before the end of the range
                    # the end length in range is the actual length of the
                    # object (end - start)
                    if end <= range_end:
                        end_length_in_range[object_pos,
                                            sim_id, param_id] = (end - start)
                    else:
                        # otherwise, the object end is after the range end
                        # and thus the end length in range is the difference of
                        # the range end and the object start
                        end_length_in_range[object_pos,
                                            sim_id, param_id] = (range_end -
                                                                start)

            object_pos += 1
        current_sim_nb += nb_processes

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


    def _copy_scripts(self, folder, path, figure_script_name, figure_script_path,
                     figure_layout_path, figure_layout_name, overwrite_script):
        """
        Function specific to Schelski et al., 2025

        Args:
            path:
            figure_script_name:
            figure_script_path:
            figure_layout_name:
            overwrite_script:

        Returns:

        """
        print(folder)
        # copy script and layout to each folder
        script_path = os.path.join(path, figure_script_name.replace(".py",
                                                                    "_" +
                                                                    folder + ".py"))

        if (not os.path.exists(script_path)) | overwrite_script:
            shutil.copy(figure_script_path, script_path)

        layout_path = os.path.join(path, figure_layout_name)
        if not os.path.exists(layout_path):
            shutil.copy(figure_layout_path, layout_path)
        new_script_path = os.path.join(path, os.path.basename(__file__))
        try:
            new_script_path_repl = new_script_path.replace(".py",
                                                           "_" + folder + ".py")
            if os.path.exists(new_script_path_repl):
                os.remove(new_script_path_repl)
            shutil.copy(__file__, new_script_path_repl)
        except:
            new_script_path_repl = new_script_path.replace(".py",
                                                           "_" + folder[
                                                                 :10] + ".py")
            if os.path.exists(new_script_path_repl):
                os.remove(new_script_path_repl)
            shutil.copy(__file__, new_script_path_repl)


    def _save_local_data(self, data_path):
        """
        Function specific to Schelski et al., 2025

        Args:
            data_path:

        Returns:

        """

        def wide_to_long_data(data, populations, columns):
            long_data = []
            for population in populations:
                print(population)
                new_data = data.copy()
                new_data["population"] = population
                for column in columns:
                    new_data[column] = new_data[column + "_" + population]
                    for drop_column in new_data.columns:
                        if drop_column.find(column + "_") != -1:
                            new_data = new_data.drop(drop_column, axis=1)
                long_data.append(new_data)
            return pd.concat(long_data).reset_index()

        data = pd.read_feather(data_path)

        metadata_path = os.path.join(path, "metadata.csv")
        metadata = pd.read_csv(metadata_path)

        new_data = pd.DataFrame()
        new_data["simulation_nb"] = data["simulation_nb"]
        if "param_nb" in data.columns:
            new_data["param_nb"] = data["param_nb"]
        for column in data.columns:
            if (column.startswith("k_") | column.startswith("v_")):
                new_data[column] = data[column]

        new_data["position"] = data[posthoc_data_analysis +
                                    "1D_density_position"]

        populations = {"labile_growing_1": "Gu",
                       "labile_pausing_2": "Pu",
                       "stable_3": "Ps",
                       "stable_growing_4": "Gs",
                       "stable_pausing_5": "Psu"
                       }
        total_src_str = posthoc_data_analysis + "1D_density_state_"
        total_target_str = "MT_density_"
        for source, target in populations.items():
            new_data[total_target_str + target] = data[
                total_src_str + source]

        # unstable_src_str = "local_unstable_density_1D_density_state_"
        # unstable_target_str = "MT_unstable_density_"
        # for source, target in populations.items():
        #     if total_target_str+target not in new_data.columns:
        #         continue
        #     if (total_src_str + source) in data.columns:
        #         new_data[total_target_str + target] = data[total_src_str + source]

        # stable_src_str = "local_stable_density_1D_density_state_"
        # stable_target_str = "MT_stable_density_"
        # for source, target in populations.items():
        #     if total_target_str+target not in new_data.columns:
        #         continue
        #     if (total_src_str + source) in data.columns:
        #         new_data[total_target_str + target] = data[total_src_str + source]

        new_data["solution_type"] = "stochastic"
        new_data["neurite_length"] = metadata["prop_position_max"].values[0]

        del data
        print("Start averaging data...")
        # start = time.time()
        group_params = [param
                        for param in new_data.columns
                        if param.startswith("k_")]

        if "param_nb" in new_data.columns:
            group_params.append("param_nb")
        # data = new_data.groupby([*group_params, "v_f", "v_g",
        #                          "position",
        #                          "solution_type"]).mean().reset_index()
        # del new_data

        data = new_data

        # print(time.time() - start)
        populations = ["Gu", "Pu", "Ps"
            , "Gs", "Psu"
                       ]
        columns = ["MT_density"]

        print("Start conversion to long data...")
        data = wide_to_long_data(data, populations, columns)

        # print("Start saving data...")
        # data.to_feather(os.path.join(processed_data_path))

        return data


    def save_simulation_data(self, stochastic_path, folder, data_folders,
                             figure_script_name, figure_script_path,
                             figure_layout_path, figure_layout_name,
                             overwrite_script, orig_reference_pos, experiment
                             ):
        one_simulation_data = pd.DataFrame()

        path = os.path.join(stochastic_path, folder)
        self._copy_scripts(folder, path, figure_script_name, figure_script_path,
                           figure_layout_path, figure_layout_name,
                           overwrite_script)

        data_path = os.path.join(path, "_data_local_density",
                                 "data.feather")
        if os.path.exists(data_path):
            data = self._save_local_data(data_path)
            data["type"] = "simulation"
            data["group"] = "populations"
            data["position"] /= data["position"].max()

            all_positions = data["position"].drop_duplicates().values
            new_reference_pos = np.sort(all_positions[all_positions >
                                                      orig_reference_pos])[
                0]

            def normalize_on_reference(data):
                data["MT_density_norm"] = data["MT_density"] / data.loc[
                    data["position"] == new_reference_pos,
                    "MT_density"]
                return data

            data = data.groupby(["group"]).apply(
                normalize_on_reference).reset_index()

            one_simulation_data = pd.concat([one_simulation_data, data])

        folders = data_folders

        for data_name, folder in folders.items():
            data_path = os.path.join(path, folder, "data.feather")

            if not os.path.exists(data_path):
                print("The data for", data_name, "(", folder,
                      ") was not saved",
                      "after the simulation.")
                continue
            data = pd.read_feather(data_path)
            metadata_path = os.path.join(path, "metadata.csv")
            metadata = pd.read_csv(metadata_path)

            position_column = [column for column in data.columns
                               if column.find("1D_density_position") != -1][
                0]
            data_columns = [column for column in data.columns
                            if ((column.find("1D_density_position") == -1)
                                & (column.find("1D_density") != -1))]

            for data_column in data_columns:
                # if len(data_column) > 1:
                #     data_column = [column for column in data_column
                #                    if column.find("Total") != -1]
                # else:
                #     data_column = data_column[0]

                new_data = pd.DataFrame()
                new_data["simulation_nb"] = data["simulation_nb"]
                if "param_nb" in data.columns:
                    new_data["param_nb"] = data["param_nb"]

                param_columns = []
                if "param_nb" in new_data.columns:
                    param_columns.append("param_nb")
                for column in data.columns:
                    if (column.startswith("k_") | column.startswith("v_")):
                        new_data[column] = data[column]
                        param_columns.append(column)

                new_data["MT_density"] = data[data_column]

                if len(data[position_column].drop_duplicates()) > 1:
                    new_data["position"] = data[position_column]
                else:
                    groups = [*param_columns,
                              "simulation_nb"]
                    new_data["position"] = new_data.groupby(groups)[
                        "simulation_nb"].transform(
                        lambda x: list(range(len(x))))
                    new_data["position"] *= data["position_max"].max() / \
                                            new_data["position"].max()

                new_data["group"] = data_name
                new_data["details"] = data_column
                new_data["position"] /= new_data["position"].max()
                new_data["time"] = data["time"]
                new_data = new_data.groupby([*param_columns,
                                             "group", "details",
                                             "position", "time",
                                             "simulation_nb"]).mean().reset_index()
                new_data["type"] = "simulation"
                all_positions = new_data[
                    "position"].drop_duplicates().values
                new_reference_pos = np.sort(all_positions[all_positions >=
                                                          orig_reference_pos])[
                    0]

                def normalize_on_reference(data):
                    all_positions = data[
                        "position"].drop_duplicates().values
                    new_reference_pos = np.sort(
                        all_positions[all_positions >=
                                      orig_reference_pos])
                    if len(new_reference_pos) == 0:
                        return data
                    new_reference_pos = new_reference_pos[0]
                    data["MT_density_norm"] = data["MT_density"] / data.loc[
                        data["position"] == new_reference_pos,
                        "MT_density"].values[0]
                    return data

                new_data = new_data.groupby(
                    [*param_columns, "group",
                     "details",
                     "simulation_nb"]).apply(
                    normalize_on_reference).reset_index(drop=True)

                one_simulation_data = pd.concat(
                    (one_simulation_data, new_data))

        if len(one_simulation_data) > 0:
            one_simulation_data.reset_index(drop=True).to_feather(
                os.path.join(path, "all_MT_distributions.feather"))

            # one_simulation_data.to_csv(
            #     os.path.join(path, "all_MT_distributions.csv"))

            group_params = [param
                            for param in one_simulation_data.columns
                            if param.startswith("k_")]
            if "param_nb" in one_simulation_data.columns:
                group_params.append("param_nb")

            one_simulation_data["group"] = one_simulation_data[
                "group"].fillna(value=0)
            if "population" not in one_simulation_data.columns:
                one_simulation_data["population"] = np.nan
            one_simulation_data["population"] = one_simulation_data[
                "population"].fillna(value=0)

            one_simulation_data = one_simulation_data.groupby(
                [*group_params, "v_f", "v_g",
                 "position", "group", "details",
                 "population", "type",
                 "time"]).mean().reset_index()

            one_simulation_data["position_um"] = one_simulation_data[
                "position"]

            one_simulation_data["position"] = one_simulation_data.groupby(
                [*param_columns,
                 "group", "details",
                 "population", "type", ])["position"].transform(
                lambda x: x / x.max())

            one_simulation_data["position"] = one_simulation_data[
                "position"].round(3)
            # normalization on averages needed. When averaging normalized single
            # simulations, the result is different than normalizing on the
            # averages. This is mainly due to
            if "level_0" in one_simulation_data.columns:
                one_simulation_data.drop("level_0", axis=1, inplace=True)

            one_simulation_data = one_simulation_data.groupby(
                [*group_params, "v_f", "v_g", "group", "details",
                 "population", "type",
                 "time"]).apply(normalize_on_reference).reset_index(drop=True)

            one_simulation_data.to_csv(os.path.join(path,
                                                    "all_MT_distributions_mean_" + experiment + ".csv"))

        return one_simulation_data

    def get_simulation_experiment_data(self, experiment,
                                       base_folder, data_folders,
                                       orig_reference_pos=0.5):
        """
        Function specific to Schelski et al., 2025

        Args:
            experiment:
            base_folder:
            data_folders: dictionary with keys being the short name of the
                data extraction and values being the folder names

        Returns:

        """

        if data_folders is None:
            data_folders = {"stable": "_data_local_stable_density",
                           "unstable": "_data_local_unstable_density",
                           "growing": "_data_local_growing_tip_density",
                           "all": "_data_local_density_all",
                           # "populations" : "_data_local_density",
                           "modification": "_data_modification_density"}
        else:
            data_folders = {key[1]: "_data_"+key[0]
                            for key, _ in data_folders.items()}

        # base_folder = "C:\\Users\\schelskim\\Nextcloud\\01ANALYSIS\\TUBB\\simulations\\"

        stochastic_path = os.path.join(base_folder,
                                       "MTRF_stochastic_steady_state")
        analyzed_path = os.path.join(base_folder, "processed_data")

        exp_MT_density_path = os.path.join(base_folder, "MT_distribution",
                                           "DIV1")
        exp_MT_density_file_name = "MT_distributions_modifications_stage2.csv"
        exp_MT_density_data = pd.read_csv(os.path.join(exp_MT_density_path,
                                                       exp_MT_density_file_name))

        # exp_MT_density_data = exp_MT_density_data.loc[exp_MT_density_data["position"] > excluded_start]
        # exp_MT_density_data["position"] -= excluded_start
        # exp_MT_density_data["position"] /= (1 - excluded_start)
        # get first position after new reference position:
        all_positions = exp_MT_density_data["position"].drop_duplicates().values
        new_reference_pos = \
        np.sort(all_positions[all_positions > orig_reference_pos])[0]
        # replace nan with -1, otherwise
        exp_MT_density_data.loc[np.isnan(
            exp_MT_density_data[
                "actub_detyr_fraction"]), "actub_detyr_fraction"] = -1
        exp_MT_density_data.loc[np.isnan(
            exp_MT_density_data[
                "tyrtub_actub_fraction"]), "tyrtub_actub_fraction"] = -1
        exp_MT_density_data.set_index(["type", "group", "channel_name",
                                       "actub_detyr_fraction",
                                       "tyrtub_actub_fraction"], inplace=True)

        # exp_MT_density_data.sort_values(["type", "group", "channel_name",
        #                                  "actub_detyr_fraction",
        #                                  "tyrtub_actub_fraction", "position"],
        #                                 inplace=True)
        # exp_MT_density_data.reset_index(inplace=True, drop=True)

        # exp_MT_density_data["MT_density_norm"] = exp_MT_density_data["MT_density"]

        # print(exp_MT_density_data.columns)

        def normalize_on_reference(data):
            data["MT_density_norm"] = data["MT_density"] / data.loc[
                data["position"] == new_reference_pos,
                "MT_density"]
            return data

        exp_MT_density_data = exp_MT_density_data.groupby(
            ["type", "group", "channel_name",
             "actub_detyr_fraction",
             "tyrtub_actub_fraction"]).apply(
            normalize_on_reference).reset_index(drop=True)
        exp_MT_density_data.reset_index(inplace=True)
        # exp_MT_density_data = pd.DataFrame()

        exp_growing_tips_path = os.path.join(base_folder,
                                             "EB_comet_analysis_new")
        exp_growing_tips_file_name = "panelB_EB_location_data_mean.csv"
        exp_growing_tips_data_orig = pd.read_csv(
            os.path.join(exp_growing_tips_path,
                         exp_growing_tips_file_name))

        exp_growing_tips_data = pd.DataFrame()
        exp_growing_tips_data["position"] = exp_growing_tips_data_orig[
            "neurite_pos_rel"]
        start_point = exp_growing_tips_data["position"].min()
        exp_growing_tips_data["position"] -= start_point
        exp_growing_tips_data["position"] /= (1 - start_point)
        exp_growing_tips_data["position"] /= exp_growing_tips_data[
            "position"].max()
        exp_growing_tips_data["MT_density"] = exp_growing_tips_data_orig[
            "number"]
        exp_growing_tips_data["type"] = "experiment"
        exp_growing_tips_data["group"] = "growing"

        # exp_growing_tips_data = pd.DataFrame()

        if not os.path.exists(analyzed_path):
            os.mkdir(analyzed_path)

        stochastic_folders = [experiment]

        exp_name = stochastic_folders[0]

        posthoc_data_analysis = ""

        overwrite_processed_data = True
        overwrite_script = False

        processed_file_name = "panelABCDEFG_data_processed" + exp_name + ".feather"

        figure_script_name = "Figure1_SimVsExp.py"
        figure_script_path = os.path.join(base_folder, figure_script_name)
        figure_layout_name = "__figure__1.csv"
        figure_layout_path = os.path.join(base_folder, figure_layout_name)

        experiment = stochastic_folders[0]
        # experiment = "MTRF_stable_growth_GupuVgMap_KcuTip5_kNS12_varGsps_axon_160um"
        # experiment = stochastic_folders[0]
        if experiment != "":
            processed_data_folder = os.path.join(analyzed_path, experiment)
            processed_data_path = os.path.join(processed_data_folder,
                                               processed_file_name)
            if not os.path.exists(processed_data_folder):
                os.mkdir(processed_data_folder)
                one_simulation_data = pd.DataFrame()
            elif not overwrite_processed_data:
                one_simulation_data = pd.read_feather(processed_data_path)
            else:
                one_simulation_data = pd.DataFrame()

            # copy script into processed path
            script_name = os.path.basename(__file__).replace(".py",
                                                             "_" + experiment + ".py")
            new_script_path = os.path.join(processed_data_folder, script_name)
            shutil.copy(__file__, new_script_path)

            script_name = figure_script_name.replace(".py",
                                                     "_" + experiment + ".py")
            new_figure_script_path = os.path.join(processed_data_folder,
                                                  script_name)
            if not os.path.exists(new_figure_script_path) | overwrite_script:
                shutil.copy(figure_script_path, new_figure_script_path)

            script_name = figure_layout_name.replace(".csv",
                                                     "_" + experiment + ".csv")
            new_figure_script_path = os.path.join(processed_data_folder,
                                                  script_name)
            if not os.path.exists(new_figure_script_path):
                shutil.copy(figure_layout_path, new_figure_script_path)


        all_data = pd.DataFrame()

        all_data = pd.concat([all_data, exp_MT_density_data])

        for folder in stochastic_folders:
            one_simulation_data = self.save_simulation_data(stochastic_path,
                                                            folder,
                                                            data_folders,
                                                            figure_script_name,
                                                            figure_script_path,
                                                            figure_layout_path,
                                                            figure_layout_name,
                                                            overwrite_script,
                                                            orig_reference_pos,
                                                            experiment)

            if len(one_simulation_data) == 0:
                continue

            # one_simulation_data["MT_density_norm"] = one_simulation_data.groupby(["group",
            #                                                 "details"])["MT_density"].transform(lambda x: x/x.iloc[0])

            all_data = pd.concat([all_data, one_simulation_data])

            one_simulation_data = pd.concat([one_simulation_data,
                                             exp_MT_density_data,
                                             exp_growing_tips_data])

            one_simulation_data = one_simulation_data.fillna(-1)

            one_simulation_data.to_csv(os.path.join(stochastic_path, experiment,
                                                    "panelAB_all_MT_distributions_SimVsExp_" + exp_name + ".csv"))

            if experiment != "":
                print(processed_data_path)
                one_simulation_data.reset_index(drop=True).to_csv(
                    processed_data_path)
                del one_simulation_data

        all_data = all_data.fillna(-1)
        all_data.to_csv(processed_data_path)

        return all_data, processed_file_name

    def get_global_object_orientation(self):
        """

        Get percentage of forward oriented objects

        Returns:

        """

        if self.simulation is not None:
            self.data_folder = self.simulation.data_folder

        object_states_array = torch.load(os.path.join(self.data_folder,
                                                "object_states.pt"))
        # else:
        #     object_states_array = self.simulation.object_states

        object_states = object_states_array[:1, 0]
        orientation = object_states_array[:1, 3]

        # if self.simulation is None:
        # orientation = torch.load(os.path.join(self.data_folder,
        #                                         "orientation.pt"))
        # else:
        #     orientation = self.simulation.orientation


        forward_orientation = 1
        orientation_forward = copy.deepcopy(orientation)
        orientation_forward[orientation_forward != forward_orientation] = 0
        orientation_count_forward = torch.count_nonzero(orientation_forward, dim=1)

        reverse_orientation = -1
        orientation[orientation != reverse_orientation] = 0
        orientation_count_reverse = torch.count_nonzero(orientation, dim=1)
        # object_count = torch.count_nonzero(object_states, dim=1)

        relative_orientation = ((orientation_count_forward /
                                 (orientation_count_forward +
                                  orientation_count_reverse)))

        orientation_data = {}
        # length_decay_data["time"] = (torch.Tensor(range(len(all_lengths_in_range)))
        #                              * time_resolution).unsqueeze(1).unsqueeze(1).expand(all_lengths_in_range.shape)
        orientation_data["fraction"] = relative_orientation.unsqueeze(1)

        all_data = {}
        all_data["orientation"] = orientation_data

        simulation.SSA._save_data(all_data, self.data_folder, 0)



    def track_local_object_length(self, range_start, range_end,
                                  max_pos = None, time_resolution = None):
        """
        Simulate a photoconversion experiment. The accuracy is limited by the
        time resolution of the simulation.

        Args:
            start_pos: relative start position of photoconversion
            end_pos: relative end position of photoconversion
            max_pos: max absolute position, only needed if simulation object
                is not defined for analyzer

        """

        orientation = None
        if self.simulation is not None:
            self.data_folder = self.simulation.data_folder

        object_states = torch.load(os.path.join(self.data_folder,
                                                "object_states.pt"))

        # orientation_path = os.path.join(self.data_folder, "orientation.pt")
        # if os.path.exists(orientation_path):
        #     orientation = torch.load(orientation_path)
        # else:
        #     orientation = np.zeros(object_states.shape[1:])
        #     print("WARNING: No orientation data was saved.")

        orientation = object_states[0,3]

        properties_array = torch.load(os.path.join(self.data_folder,
                                                   "property_array.pt"))

        print(orientation.shape, object_states.shape)

        if (max_pos is None) & (self.simulation is None):
            raise ValueError("For the photoconversion simulation, either the "
                             "simulation object or the max_pos need to be "
                             "defined.")
        elif max_pos is None:
            max_pos = self.simulation.properties[0].max_value

        # if (time_resolution is None) & (self.simulation is None):
        #     raise ValueError("For the photoconversion simulation, either the "
        #                      "simulation object or the time_resolution "
        #                      "need to be defined.")
        # elif time_resolution is None:
        #     time_resolution = self.simulation.time_resolution

        # get absolute start and end pos
        range_start_abs = range_start * max_pos
        range_end_abs = range_end * max_pos

        # For each timepoint have the object length that was initially in the
        # defined compartment range (between start_pos and end_pos)

        all_lengths_in_range = []
        # For first timepoint:
        # Go through each object and check how much length is
        # between defined start and end position in compartment.
        # Save start and end position of object that is in range,
        # in separate array
        first_timepoint_idx_states = 4
        initial_creation_time = object_states[1, first_timepoint_idx_states]
        first_timepoint_idx_props = 1
        properties = properties_array[first_timepoint_idx_props]

        # first_false_position = torch.count_nonzero(
        #     object_states[0, first_timepoint_idx_states], dim=0)
        # max_nb_objects = first_false_position.max()

        start_length_in_range = np.zeros(object_states.shape[2:])
        end_length_in_range = np.zeros(object_states.shape[2:])

        def to_cuda(x):
            return numba.cuda.to_device(np.ascontiguousarray(x))

        start_length_in_range = to_cuda(start_length_in_range)
        end_length_in_range = to_cuda(end_length_in_range)

        nb_parallel_cores = 32

        nb_SM, nb_cc = simulation.SSA._get_number_of_cuda_cores()

        properties_sum = 0
        for property_nb in range(1, properties.shape[0]):
            properties_sum = properties_sum + properties[property_nb]

        # print(start_length_in_range)
        # print(end_length_in_range)
        # print(properties[0])
        # print(properties_sum)
        # print(orientation)

        _object_length_in_range[nb_SM, nb_cc](start_length_in_range,
                                              end_length_in_range,
                                              range_start_abs, range_end_abs,
                                              # 0, int(max_nb_objects),
                                              0, object_states.shape[2],
                                              to_cuda(properties[0]),
                                              to_cuda(properties_sum),
                                              to_cuda(orientation),
                                              nb_parallel_cores)

        start_length_in_range = torch.Tensor(start_length_in_range.copy_to_host()).cpu()
        end_length_in_range = torch.Tensor(end_length_in_range.copy_to_host()).cpu()

        # numba cuda kernel to get the start and end pos in range for each
        # object
        total_length_in_range = (end_length_in_range -
                                 start_length_in_range).sum(axis=0)
        all_lengths_in_range.append(torch.Tensor(total_length_in_range).unsqueeze(0))

        torch.set_printoptions(linewidth=2000)
        # print(initial_creation_time)
        # print(object_states[1, first_timepoint_idx_states][:8,0,0])
        # print(object_states[0, first_timepoint_idx_states])

        object_mask = initial_creation_time == 0
        # For following timepoints:
        # If same object is still present (check position and creation time)
        # check how much the length is below the end point in range.
        # Subtract that amount from the length of the first timepoint.
        for timepoint in range(1, object_states.shape[1] -
                                  first_timepoint_idx_states):

            timepoint_idx_states = timepoint + first_timepoint_idx_states
            timepoint_idx_props = timepoint + first_timepoint_idx_props
            creation_time = object_states[1, timepoint_idx_states]
            # update mask to only include the same objects as in the beginning
            # (indicated by same position in array and same creation time)
            object_mask = object_mask | (creation_time != initial_creation_time)

            # print(timepoint, (~object_mask).sum()/object_mask.sum())

            # initial_creation_time = object_states[1, timepoint_idx_states]
            # print(timepoint_idx_states, object_states[0, timepoint_idx_states][:8,0,0])
            # print(timepoint_idx_states,
            #       properties_array[timepoint_idx_states, 1][:8,0,0])
            # print(timepoint_idx_states,
            #       properties_array[timepoint_idx_states, 2][:8,0,0])
            # print(object_mask.shape)

            properties = properties_array[timepoint_idx_props]
            # set all properties for initial objects that are not there anymore
            # to 0. Thereby end_length_in_range is also reduced to 0 and no length
            # for that object is considered
            properties_sum = 0
            for property_nb in range(1, properties.shape[0]):
                properties_sum = properties_sum + properties[property_nb]

            properties_sum[object_mask] = 0
            # reduce end_pos in range if properties_sum is below it
            end_length_in_range[properties_sum <
                                end_length_in_range] = properties_sum[properties_sum <
                                                                      end_length_in_range]
            # if the property sum is below the start pos in range, the end pos
            # is set to 0, since nothing of the object in the initial range is
            # left
            end_length_in_range[properties_sum <= start_length_in_range] = 0
            start_length_in_range[properties_sum <= start_length_in_range] = 0
            # sum total length in range for all objects
            # print((end_length_in_range - start_length_in_range).max())

            total_length_in_range = (end_length_in_range -
                                     start_length_in_range).sum(axis=0)
            all_lengths_in_range.append(torch.Tensor(total_length_in_range).unsqueeze(0))

        all_lengths_in_range = torch.concat(all_lengths_in_range, axis=0).unsqueeze(1)
        length_decay_data = {}
        # length_decay_data["time"] = (torch.Tensor(range(len(all_lengths_in_range)))
        #                              * time_resolution).unsqueeze(1).unsqueeze(1).expand(all_lengths_in_range.shape)
        length_decay_data["decay"] = all_lengths_in_range

        all_data = {}
        all_data["length_decay_"
                 +str(range_start)+"-"+str(range_end)] = length_decay_data

        simulation.SSA._save_data(all_data, self.data_folder, 0)

        # return all_data

    def get_local_stable_unstable_dynamic_properties(self, range_starts,
                                                     range_ends, length,
                                                     time_res,
                                                     max_rate_stable_MTs_one_population,
                                                     max_rate_stable_MTs,
                                                     min_rate_unstable_MTs,
                                                     range_names=None,
                                                     max_time_to_analyze=40,
                                                     length_groups=None,
                                                     min_decay_rates_diff=2):
        """

        Args:
            range_starts: list of starts of ranges to analyze locally
            range_ends: list of ends of ranges
            length: length of neurite in simulation
            time_res: time resolution to use for analysis
            max_rate_stable_MTs_one_population: if there is only one rate,
                everything above this threshold is considered unstable,
                everything below is considered stable
            max_rate_stable_MTs: if there are two different rates, then this
                threshold will be used to check whether both rates are for
                unstable MTs
            min_rate_unstable_MTs: f there are two different rates, then this
                 threshold will be used to check whether both rates are for
                 unstable MTs
            range_names: list of strings for names to be used for defined ranges
                in range_starts and range_ends. Needs to be same length as
                range_starts/range_ends.
            max_time_to_analyze: max time to analyze to get decay constants
            length_groups: groups of neurite lengths to analyze
            min_decay_rates_diff: minimum difference of the slow and fast decay
                rates to be considered decay rates from separate propulations.
                Just randomly decay rates can differ to some extent, even if
                they come from the same population of objects.

        Returns:

        """

        if range_names is not None:
            if len(range_names) != len(range_starts):
                raise ValueError("If range_names is defined, then it has to be"
                                 " the same length as range_starts/range_ends.")
        if len(range_starts) != len(range_ends):
            raise ValueError("The length of range_starts has to be the same"
                             " as the length of range_ends.")
        self.get_local_decay(range_starts, range_ends, length, time_res)

        if length_groups is None:
            length_groups = [10,60]

        if self.data_folder is not None:
            base_path = self.data_folder
        else:
            base_path = self.simulation.data_folder

        file_name = "MT_decay_data_model.feather"
        data_path = os.path.join(base_path, file_name)
        if not os.path.exists(data_path):
            raise ValueError("There was a problem when analyzing "
                             " local MT decay or the corresponding file"
                             "MT_decay_data_model.feather can't be found in"
                             " the simulation folder. ")
        data = pd.read_feather(os.path.join(base_path, file_name))

        exp_name = os.path.basename(base_path)

        # data = data.loc[data["simulation_nb"] < 400]

        max_time = max_time_to_analyze

        data["time"] -= data["time"].min()
        data = data.loc[data["time"] <= max_time]

        # first_timepoint = data.loc[data["time"] == 0]
        # high_int = first_timepoint.loc[first_timepoint["total"] > 16]
        # high_int.set_index(["simulation_nb", "range"], inplace=True)

        # data.set_index(["simulation_nb", "range"], inplace=True)
        # data = data.loc[high_int.index.values].reset_index()

        folder = base_path
        data["neurite_length_um"] = 40

        def remove_early_vals(data):
            vals = data["total_norm"].values
            vals[np.isnan(vals)] = 0
            first_index = np.argmax(vals)
            data = data.iloc[first_index:]
            return data

        data["total_norm"] = data.groupby(["simulation_nb",
                                           "range"])["total"].transform(
            lambda x: x / x.max())


        all_cat_rates = []
        sb.set(font_scale=2)
        for data_nb, data in enumerate([data]):
            def fit_MT_cat_rates(data, mono_exponential,
                                 dual_exponential, result_index, data_nb=0,
                                 get_fitted_data=False):
                # print(data)
                # print(data.columns, data.reset_index().columns)
                number_of_simulations = len(
                    data.reset_index()[["simulation_nb"]].drop_duplicates())
                print("Analyzing simulation ",
                      data[["simulation_nb"]].drop_duplicates().values[0])

                data_mean = data
                # data_mean = data.groupby(["time"]).mean().reset_index()
                # data = data.loc[data["total_norm"] > 0]
                if ((data_mean.loc[data_mean["time"] == 0, "total_norm"].values[
                         0] == 0) |
                        (np.isnan(data_mean.loc[data_mean[
                                                    "time"] == 0, "total_norm"].values[
                                      0]))):
                    return pd.DataFrame()
                fit = scipy.optimize.curve_fit(dual_exponential,
                                               data_mean["time"],
                                               data["total_norm"],
                                               bounds=(0, np.inf),
                                               maxfev=10000)
                # print(fit)

                # fit_mono = scipy.optimize.curve_fit(mono_exponential, data["time"],
                #                                data["total_norm"],
                #                                bounds=(0, np.inf),
                #                                maxfev=10000)
                # if len(fit[0]) > 2:
                exponential_fit_cell = functools.partial(dual_exponential,
                                                         m=fit[0][0],
                                                         a=fit[0][1],
                                                         n=fit[0][2],
                                                         b=fit[0][3])
                # else:
                # mono_exponential_fit_cell = functools.partial(mono_exponential,
                #                                               m=fit[0][0], a=fit[0][1])

                if len(fit[0]) > 2:
                    # plt.title("DUAL EXP: " + str(np.round(fit[0][0],2))+"-"+
                    #           str(np.round(fit[0][1],2))+"-"+
                    #           str(np.round(fit[0][2],2))+"-"+
                    #           str(np.round(fit[0][3],2))+"-")
                    rates = np.array(
                        [fit[0][result_index + 0], fit[0][result_index + 2]])
                    low_index = np.where(rates == rates.min())
                    high_index = np.where(rates == rates.max())
                    fractions = np.array([fit[0][0], fit[0][2]])
                    low_fraction = fractions[low_index][0]
                    high_fraction = fractions[high_index][0]
                    low_fraction = low_fraction / (low_fraction + high_fraction)
                    high_fraction = high_fraction / (
                                low_fraction + high_fraction)
                    rates_df = pd.DataFrame(columns=("low", "high",
                                                     "Low fraction",
                                                     "High fraction",
                                                     "n_cells"),
                                            data=[[rates.min(), rates.max(),
                                                   low_fraction, high_fraction,
                                                   number_of_simulations]],
                                            index=[0])

                # if rates.max() > 2:
                # sb.set(font_scale=2)
                data_fit = exponential_fit_cell(data_mean["time"])
                # mono_data_fit = mono_exponential_fit_cell(data["time"])

                if get_fitted_data:
                    dual_exp_data = data_mean.copy()
                    dual_exp_data["total_norm"] = data_fit
                    dual_exp_data["type"] = "dual_exp"

                    # mono_exp_data = data.copy()
                    # mono_exp_data["total_norm"] = mono_data_fit
                    # mono_exp_data["type"] = "mono_exp"

                    data_mean["type"] = "data"

                    all_data = pd.concat([data_mean, dual_exp_data])
                    return all_data

                # name = "Dual_vs_mono_exponential_"+str(result_index)+"_"+str(data_nb)
                # plt.figure()
                # plt.gca().set_facecolor('white')
                # plt.gca().grid(True, color='grey', linestyle='-', linewidth=1)
                # ax = plt.plot(data["time"], data["total_norm"], color="black", linewidth=3)
                # ax = plt.plot(data["time"], mono_data_fit, color="grey", linewidth=3)
                # ax = plt.plot(data["time"], data_fit, color="orange", linewidth=3)
                # plt.gca().set_facecolor('white')
                # plt.gca().grid(True, color='grey', linestyle='-', linewidth=1)
                # plt.ylim([-0.02, 1.1])
                # plt.xlabel("Time [min]")
                # plt.ylabel("Relative microtubule mass")
                # # title = str(idx)
                # # plt.title(title)
                if len(fit[0]) > 2:

                    return rates_df

            def dual_exponential(x, m, a, n, b):
                return m * np.exp(-a * x) + n * np.exp(-b * x)

            def mono_exponential(x, m, a):
                return m * np.exp(-a * x)

            # print(data.columns)
            # print(len(data[["date", "exp", "cell"]].drop_duplicates()))

            all_cat_rates_exp = []

            for nb, length in enumerate(length_groups[1:]):
                length_data = data.loc[
                    (data["neurite_length_um"] > length_groups[nb]) &
                    (data["neurite_length_um"] <= length)]
                print("\n Length group: ", length)
                length_data["length_group"] = length

                cat_rates_mean = None
                all_cat_rates_length = []
                cat_rates_mean = length_data.groupby(
                    ["simulation_nb", "range", "k_cs"]).apply(fit_MT_cat_rates,
                                                              mono_exponential,
                                                              dual_exponential,
                                                              result_index=1,
                                                              data_nb=data_nb)

                # fitted_data = length_data.groupby(["simulation_nb", "range", "k_cs"]).apply(fit_MT_cat_rates,
                #                                                                   mono_exponential, dual_exponential,
                #                                                                   result_index=1,
                #                                                                   data_nb=data_nb,
                #                                                                   get_fitted_data=True)

                # all_cat_rates_length = pd.concat(all_cat_rates_length)
                cat_rates_mean["neurite_length"] = length
                all_cat_rates_exp.append(cat_rates_mean)

            all_cat_rates_exp = pd.concat(all_cat_rates_exp)
            all_cat_rates_exp["exp_type"] = data_nb
            all_cat_rates.append(all_cat_rates_exp)

        all_cat_rates = pd.concat(all_cat_rates).reset_index()
        # cat_rates_60s = all_cat_rates.loc[all_cat_rates["exp_type"] == 1]
        # print(all_cat_rates.groupby(["neurite_length", "exp_type"]).mean())

        if range_names is None:
            range_names = [str(idx) for idx in range(len(range_starts))]

        for pos_index in range(len(range_starts)):
            range_start = range_starts[pos_index]
            range_end = range_ends[pos_index]
            range_str = str(range_start) + "-" + str(range_end)
            all_cat_rates.loc[all_cat_rates["range"] == range_str,
                               "position"] = range_names[pos_index]

        all_cat_rates["position"] = "far"
        all_cat_rates.loc[
            all_cat_rates["range"] == "0.25-0.35", "position"] = "close"

        cat_rates = all_cat_rates[["simulation_nb",
                                   "exp_type", "neurite_length",
                                   "high", "low", "High fraction",
                                   "Low fraction", "n_cells", "position"]]

        cat_rates["source"] = "simulation"

        # fitted_data.drop("simulation_nb", axis=1, inplace=True)
        # fitted_data = fitted_data.drop("k_cs", axis=1).reset_index()
        # fitted_data["source"] = "simulation"
        # fitted_data["position"] = "far"
        # fitted_data.loc[fitted_data["range"] == "0.25-0.35", "position"] = "close"
        # fitted_data.to_csv(os.path.join(base_path, "MT_decay_sim_close_far_fitted_"+exp_name+".csv"))

        cat_rates.to_csv(
            os.path.join(base_path, "MT_decay_close_far_sim_" + exp_name +
                         "_maxt" + str(max_time) + "_raw.csv"))

        cat_rates["one_population"] = 0
        cat_rates["diff"] = cat_rates["high"] / cat_rates["low"]
        cat_rates.loc[cat_rates["diff"] < min_decay_rates_diff,
                      "one_population"] = 1
        # cat_rates.loc[cat_rates["high"] > 5, "one_population"] = 1

        cat_rates["rate_weighted_mean"] = ((cat_rates["High fraction"] *
                                            cat_rates["high"])
                                           + (cat_rates["Low fraction"] *
                                              cat_rates["low"]))

        cat_rates["only_stable"] = 0
        cat_rates["only_unstable"] = 0

        # if only one population of MTs is present and its lifetime is similar to
        # the lifetime of stable MTs, all MTs are stable
        cat_rates.loc[(cat_rates["one_population"] == 1) &
                      (cat_rates[
                           "low"] < max_rate_stable_MTs_one_population), "only_stable"] = 1

        # if only one population of MTs is present and its lifetime is similar to
        # the lifetime of unstable MTs, all MTs are unstable
        cat_rates.loc[(cat_rates["one_population"] == 1) &
                      (cat_rates[
                           "low"] > max_rate_stable_MTs_one_population), "only_unstable"] = 1

        # if the higher rate is as low as the rate for stable MTs, then all MTs
        # are stable
        cat_rates.loc[(cat_rates["one_population"] == 0) &
                      (cat_rates[
                           "high"] < min_rate_unstable_MTs), "only_stable"] = 1
        # if the lower rate is higher than the max rate for stable MTs, then all MTs
        # are unstable
        cat_rates.loc[(cat_rates["one_population"] == 0) &
                      (cat_rates[
                           "low"] > max_rate_stable_MTs), "only_unstable"] = 1

        # set fractions correctly if there are only stable MTs
        cat_rates.loc[cat_rates["only_stable"] == 1, "High fraction"] = 0
        cat_rates.loc[cat_rates["only_stable"] == 1, "Low fraction"] = 1

        # set fractions correctly if there are only unstable MTs
        cat_rates.loc[cat_rates["only_unstable"] == 1, "High fraction"] = 1
        cat_rates.loc[cat_rates["only_unstable"] == 1, "Low fraction"] = 0

        # if there are only unstable MTs, set the unstable rate as the weighted mean
        # of both rates (weighted by the initial fraction)
        cat_rates.loc[cat_rates["only_unstable"] == 1, "high"] = cat_rates.loc[
            cat_rates["only_unstable"] == 1, "rate_weighted_mean"]

        # if there are only stable MTs, set the stable rate as the weighted mean
        # of both rates (weighted by the initial fraction)
        cat_rates.loc[cat_rates["only_stable"] == 1, "low"] = cat_rates.loc[
            cat_rates["only_stable"] == 1, "rate_weighted_mean"]

        cat_rates["stable/unstable"] = (cat_rates["Low fraction"] /
                                        cat_rates["High fraction"])

        cat_rates_close = cat_rates.loc[cat_rates["position"] == "close"]
        cat_rates_close["position"] = "far/close"
        cat_rates_far = cat_rates.loc[cat_rates["position"] == "far"]
        cat_rates_far["position"] = "far/close"

        cat_rates_close.set_index(["source", "simulation_nb", "position"],
                                  inplace=True)
        cat_rates_far.set_index(["source", "simulation_nb", "position"],
                                inplace=True)

        cat_rates_ratio = (cat_rates_far - cat_rates_close) / (
                    cat_rates_close + cat_rates_far)

        # cat_rates = pd.concat([cat_rates, cat_rates_ratio.reset_index()])

        cat_rates.to_csv(
            os.path.join(base_path, "MT_decay_close_far_sim_" + exp_name +
                         "_maxt" + str(max_time) + ".csv"))

        cat_rates_low = cat_rates.copy()
        cat_rates_low["fraction"] = cat_rates["Low fraction"]
        cat_rates_low["rate"] = cat_rates["low"]
        cat_rates_low["type"] = "stable"

        cat_rates_high = cat_rates.copy()
        cat_rates_high["fraction"] = cat_rates["High fraction"]
        cat_rates_high["rate"] = cat_rates["high"]
        cat_rates_high["type"] = "unstable"

        # cat_rates_ratio = cat_rates.copy()
        cat_rates_ratio["fraction"] = cat_rates_ratio["stable/unstable"]
        cat_rates_ratio["rate"] = cat_rates_ratio["diff"]
        cat_rates_ratio["type"] = "stable/unstable"

        cat_rates = pd.concat(
            [cat_rates_low, cat_rates_high, cat_rates_ratio.reset_index()])
        cat_rates.drop(["High fraction", "Low fraction",
                        "high", "low", "stable/unstable"], axis=1, inplace=True)

        cat_rates.to_csv(
            os.path.join(base_path, "panelA_MT_decay_close_far_sim_long_"
                         + exp_name + "_maxt" + str(max_time) + ".csv"))
        return cat_rates


    def get_local_decay(self, range_starts, range_ends, length, time_res):
        """

        Args:
            range_starts: List of relative start x points for ranges
            range_ends: List of relative start x points for ranges
            length: total length of simulation
            time_res: time resolution at which analysis should be done

        Returns:

        """

        if self.data_folder is not None:
            folder = self.data_folder
        else:
            folder = self.simulation.data_folder

        all_data = []
        for range_nb in range(len(range_starts)):
            range_start = range_starts[range_nb]
            range_end = range_ends[range_nb]

            data_name = "length_decay_" + str(range_start) + "-" + str(
                range_end)
            data_folder = "_data_" + data_name

            data_path = os.path.join(folder, data_folder)
            if os.path.exists(data_path):
                shutil.rmtree(data_path)

            self.track_local_object_length(range_start=range_start,
                                             range_end=range_end,
                                             max_pos=length)
            self.start(time_resolution=time_res, max_time=length)

            length_decay_data = pd.read_feather(os.path.join(data_path,
                                                             "data.feather"))
            length_decay_data["range"] = str(range_start) + "-" + str(range_end)
            length_decay_data["total"] = length_decay_data[data_name + "_decay"]
            all_data.append(length_decay_data)

        all_data = pd.concat(all_data)
        all_data.to_feather(os.path.join(folder, "MT_decay_data_model.feather"))

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
        if self.simulation is not None:
            self.data_folder = self.simulation.data_folder
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
        # else:
        #     all_times = self.simulation.times.to(self.device)

        print("Finished loading time files.")
        parameters = self._load_parameters()
        time_array = all_times
        # go through each folder that contains sub data
        # There is one folder for each function in DataExtraction used.
        # Time and parameter data is in the parent folder and therefore does
        # not need to be looked at per folder (therefore before this for loop)
        if self.simulation is not None:
            self.data_folder = self.simulation.data_folder
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
        # else:
        #     all_data_dict = {}
        #     for data_name, data in self.simulation.all_data.items():
        #         data_path = os.path.join(self.data_folder, "_data_"+data_name)
        #         if not os.path.exists(data_path):
        #             os.mkdir(data_path)
        #         all_data_dict[data_path] = {}
        #         for keyword, data_array in data.items():
        #             all_data_dict[data_path][keyword] = data_array.cpu()

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

        data_concat = torch.concat([data_concat, *new_data_list], dim=-1)

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

            data_shape = list(data_shapes[0])

            # check if data shape has more parameters than parameters allow
            # this is a legacy bug
            nb_parameter_vals = list(parameters.values())[0].shape[-1]

            too_many_param_vals_in_data_shape = False
            if data_shape[-1] > nb_parameter_vals:
                print("WARNING: The data contains "
                      f"{data_shape[-1] - nb_parameter_vals} more parameter "
                      f"values than expected from the defined parameter values. "
                      f"The data corresponding to the superfluous last "
                      f"parameter values was removed.")
                too_many_param_vals_in_data_shape = True
                data_shape[-1] = nb_parameter_vals

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

            data_values.append(time_array[:data_shape[0],:,:,:1].cpu().expand(data_shape).flatten().unsqueeze(1))
            column_names.append("time")
            # add information about simulation number
            simulation_array_shape = [1 for _ in data_shape]
            simulation_array_shape[2] = data_shape[2]
            simulation_array = torch.arange(data_shape[2]).view(
                simulation_array_shape)
            data_values.append(simulation_array.cpu().expand(
                data_shape).flatten().unsqueeze(1))
            column_names.append("simulation_nb")

            # the last dimensions is for the param index
            simulation_array_shape = [1 for _ in data_shape]
            simulation_array_shape[-1] = data_shape[-1]
            param_nb_array = torch.arange(data_shape[-1]).view(
                simulation_array_shape)
            data_values.append(param_nb_array.cpu().expand(
                data_shape).flatten().unsqueeze(1))
            column_names.append("param_nb")

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

            # if there are too many parameter values in the data, remove the
            # data for the last parameter value, this is a legacy bug
            # for some older simulations
            if too_many_param_vals_in_data_shape:
                data_slice = [slice(None) for _ in range(len(data_shape))]
                data_slice[-1] = slice(nb_parameter_vals)

            # now add all data
            for column_name, data in data_dict.items():
                # remove data corresponding to undefined parameter values
                if too_many_param_vals_in_data_shape:
                    data = data[data_slice]
                data_values.append(data.cpu().expand((data.shape[0],
                                                      *data_shape[1:]))
                                   .flatten().unsqueeze(1))
                column_names.append(column_name)

            data = torch.concat(data_values, dim=1)

            dataframe = pd.DataFrame(data=data, columns=column_names)
            # file_name = "_".join(column_names[1:])
            file_name = "data"
            #print("Saving dataframe...")
            dataframe.to_feather(os.path.join(data_folder,
                                              file_name + ".feather"))