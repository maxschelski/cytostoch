import torch
import numpy as np
import time

class PropertyGeometry():

    def __init__(self, properties, operation):
        """
        Define geometries of object properties by letting min or max values
        of one property depend on other properties

        Args:
            properties (list of ObjectProperty objects): List of properties that
                defines the input to the operation function
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
        if len(properties) > 1:
            raise ValueError(f"The PropertyGeometry operation "
                             f"'same_dimension_forward' is only implemented for "
                             f"one 'ObjectProperty' object in the 'properties' "
                             f"parameter. Instead {len(properties)} were "
                             f"supplied.")
        if type(properties[0].max_value) not in [float, int]:
            raise ValueError(f"For the PropertyGeometry operation "
                             f"'same_dimension_forward', only an ObjectProperty"
                             f" in the parameter 'properties' that has the "
                             f"max_value defined as float or int is allowed. "
                             f"Instead the max_value is "
                             f"{properties[0].max_value}.")
        property_array = properties[0].array
        return properties[0].max_value - property_array

    def get_limit(self):
        return self.operation(self.properties)

class Dimension():

    def __init__(self, position, length=None, direction=None):
        """

        Args:
            position (ObjectProperty):
            length (ObjectProperty):
            direction (ObjectProperty):
        """
        self.position = position
        self.length = length
        self.direction = direction

class DataExtraction():
    """
    Compress data for keeping and saving.
    Define which properties should be kept and how they should be rearranged.
    So far, this is only implemented for a 1D system modeled in 2D
    (object start position and object length).
    """

    def __init__(self, dimensions, operation, **kwargs):
        """
        Define how many dimensions in the system

        Args:
            dimensions (list of Dimension objects):
            operation (func or string): operation used to transform data for
                extraction. Name of already implemented function as string.
                Alternatively, user-defined function (encouraged to submit
                pull request to add to package) using parameters dimensions and
                resolution.
            extract_state (bool): Whether to extract object state information
        """
        self.dimensions = dimensions
        self.kwargs = kwargs

        implemented_operations = {}
        new_operation = self._operation_2D_to_1D_density
        implemented_operations["2D_to_1D_density"] = new_operation
        implemented_operations["raw"] = self._operation_raw

        if type(operation) == str:
            if operation not in implemented_operations:
                raise ValueError(f"For DataExtraction only the following"
                                 f" operations are implemented and can be "
                                 f"refered to by name in the 'operation' "
                                 f"paramter: "
                                 f"{', '.join(implemented_operations.keys())}."
                                 f" Instead the following name was supplied: "
                                 f"{operation}.")
            self.operation = implemented_operations[operation]
        else:
            self.operation = operation

    def extract(self):
        return self.operation(self.dimensions, **self.kwargs)

    def _operation_raw(self, dimensions):
        data_dict = {}
        data_dict["position"] = dimensions.position.array
        data_dict["length"] = dimensions.length.array
        return data_dict

    def _operation_2D_to_1D_density(self, dimensions, resolution=0.2,
                                    **kwargs):
        """
        Create 1D density array from start and length information without
        direction.
        Args:
            dimensions (list of Dimension objects):
            resolution (float): resolution in dimension for data export

        Returns:

        """

        if len(dimensions) > 1:
            return ValueError(f"The operation '2D_to_1D_density' is only "
                              f"implemented for 1 dimension. DataExtraction "
                              f"received {len(dimensions)} dimensions instead.")

        # create boolean data array later by expanding each microtubule in space
        # size of array will be:
        # (max_position of neurite / resolution) x nb of microtubules
        min_position = dimensions[0].position.min_value
        if min_position is None:
            min_position = 0
        max_position = dimensions[0].position.max_value

        position_dimension = int((max_position - min_position)
                                 // resolution)

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
        objects_array = ~torch.isnan(dimensions[0].position.array)
        positions_object = torch.nonzero(objects_array)
        min_pos_with_object = positions_object[:,0].min()

        position_start = dimensions[0].position.array
        max_nb_objects = position_start.shape[0]
        position_start = position_start[min_pos_with_object:max_nb_objects]

        # transform object properties into multiples of resolution
        # then transform length into end position
        position_start = torch.div(position_start, resolution,
                                   rounding_mode="floor").to(torch.short)
        position_start = torch.unsqueeze(position_start, 0)

        position_end = dimensions[0].length.array
        position_end = position_end[min_pos_with_object:max_nb_objects]
        position_end = (torch.div(position_end, resolution,
                                 rounding_mode="floor") +
                        position_start).to(indices_dtype)

        # remove negative numbers to allow uint8 dtype
        position_start[position_start < 0] = 0
        position_start = position_start.to(indices_dtype)

        # create indices array which each number
        # corresponding to one position in space (in dimension 0)
        indices = np.linspace(0,position_dimension, position_dimension+1)
        indices = np.expand_dims(indices,
                                 tuple(range(1,len(position_start.shape))))
        indices = indices_tensor(indices)

        # create boolean data array later by expanding each microtubule in space
        # use index array to set all positions in boolean data array to True
        # that are between start point and end point
        data_array = ((indices.expand(-1, *position_start.shape[1:])
                       >= position_start) &
                      (indices.expand(-1, *position_start.shape[1:])
                       <= position_end))

        # then sum across microtubules to get number of MTs at each position
        data_array = torch.sum(data_array, dim=1,
                               dtype=torch.int16)

        positions = torch.arange(min_position, max_position+resolution*0.9,
                                 resolution)
        dimensions = [-1] + [1] * (len(data_array.shape) - 1)
        positions = positions.view(*dimensions)

        data_dict = {}
        data_dict["1D_density_position"] = positions
        data_dict["1D_density"] = data_array

        return data_dict

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
        self.name = "property_"+name

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
        self.array = torch.HalfTensor([])

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

    def __init__(self, object_property, operation, values, states = None,
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
            values (1D iterable): values that should be used for the action.
                For each value, a separate set of simulations will be run.
            states (list of State objects): States in which action will be
                executed. If None, will be executed on all non zero states
            name (String): Name of action, used for data export and readability
        """
        self.states = states
        self.values = torch.HalfTensor(values)
        self.object_property = object_property
        if name != "":
            self.name = "action_"+name
        else:
            self.name = "action_state"+str(states)+object_property.name+"_OP"+str(operation)

        implemented_operations = {}
        implemented_operations["add"] = self._operation_add
        implemented_operations["subtract"] = self._operation_subtract

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

        # define variables which will be used in simulation
        self.value_array = torch.HalfTensor([])

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
        self.name = "state_"+name

        if type(self.initial_condition) is not list:
            self.initial_condition = torch.ShortTensor([self.initial_condition])

        # set variable to treat initial condtions as other simulation parameters
        self.values = self.initial_condition

class ObjectRemovalCondition():
    """
    Define conditions to remove objects from simulations - e.g. if they move
    out of the region of interest. In contrast, min and max values of properties
    prevent objects from moving out.
    """

    def __init__(self, object_properties, operation, threshold):
        """

        Args:
            object_properties (list of ObjectProperties):
            operation (function or string): If function, has to take parameters:
                object_properties and threshold. It should output a mask of
                objects that should be removed
            threshold (float): Value of last not removed object for operation
        """
        self.object_properties = object_properties
        implemented_operations = {}
        new_operation = self.operation_sum_smaller_than
        implemented_operations["sum_smaller_than"] = new_operation
        if type(operation) == str:
            if operation not in implemented_operations:
                raise ValueError(f"For ObjectRemovalCondition only the following"
                                 f" operations are implemented and can be "
                                 f"refered to by name in the 'operation' "
                                 f"paramter: "
                                 f"{', '.join(implemented_operations.keys())}."
                                 f" Instead the following name was supplied: "
                                 f"{operation}.")
            self.operation = implemented_operations[operation]
        else:
            self.operation = operation
        self.threshold = threshold

    def operation_sum_smaller_than(self, object_properties, threshold):
        sum_of_properties = torch.zeros_like(object_properties[0].array,
                                             dtype=torch.half)
        for object_property in object_properties:
            sum_of_properties += object_property.array
        return sum_of_properties < threshold

    def get_objects_to_remove(self):
        return self.operation(self.object_properties, self.threshold)


class StateTransition():

    def __init__(self, start_state=None, end_state=None, rates=None, lifetimes=None,
                 time_dependency = None, name=""):
        """

        Args:
            start_state (State object): Name of state at which the transition
                starts, If None, then the end_state is created (from state 0)
            end_state (State object): Name of state at which the transition ends
                If None, then the start_date is destroyed (to state 0)
            rates (Iterable): 1D Iterable (list, numpy array) of all rates which
                should be used; rates or lifetimes need to be defined
            lifetimes (Iterable): 1D Iterable (list, numpy array) of all
                lifetimes which should be used, will be converted to rates;
                rates or lifetimes need to be defined
            time_dependency (func): Function that takes a torch tensor
                containing the current timepoints as input and converts it to a
                tensor of factors (using only pytorch functions)
                to then be multiplied with the rates. It is recommended to have
                the time dependency function range from 0 to 1, which makes the
                supplied rates maximum rates.
            name (str): Name of transition, used for data export and readability
        """
        if (rates is None) & (lifetimes is None):
            return ValueError("Rates or lifetimes need to be specified for "
                              "the state transition from state"
                              f"{start_state} to state {end_state}.")
        self.start_state = start_state
        self.end_state = end_state
        lifetime_to_rates_factor = torch.log(torch.FloatTensor([2]))
        if rates is None:
            self.rates = lifetime_to_rates_factor / lifetimes
        else:
            self.rates = torch.HalfTensor(rates)
        self.values = self.rates
        self.time_dependency = time_dependency
        self.name = name

        # initialize variable that will be filled during simulation
        self.value_array = torch.HalfTensor([])
        self.simulation_mask = torch.HalfTensor([])
        self.transition_positions = torch.HalfTensor([])