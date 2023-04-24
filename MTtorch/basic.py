import torch

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
        if type(properties[0].max_value) is not in [float, int]:
            raise ValueError(f"For the PropertyGeometry operation "
                             f"'same_dimension_forward', only an ObjectProperty "
                             f"in the parameter 'properties' that has the "
                             f"max_value defined as float or int is allowed. "
                             f"Instead the max_value is "
                             f"{properties[0].max_value}.")
        propert_value_array = properties[0].value_array
        propert_values = propert_value_array[~propert_value_array.isnan]
        return properties[0].max_value - propert_values

    def get_limit(self):
        return self.operation(self.properties)


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
        self._name = name

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
        self.value_array = torch.HalfTensor([])

    @property
    def min_value(self):
        print(type(self._min_value))
        if type(self._min_value) == "PropertyGeometry":
            return self._min_value.get_limit()
        return self._min_value

    @property
    def max_value(self):
        if type(self._max_value) == "PropertyGeometry":
            return self._max_value.get_limit()
        return self._max_value

    @property
    def start_value(self):
        if type(self._start_value) == list:
            current_start_value = []
            for value in self._start_value:
                if type(value) == "PropertyGeometry":
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
            operation (func): function that takes the property tensor,
                a tensor of the values (same shape as the property tensor) and
                the time tensor and then outputs the transformed property tensor
                the time tensor and then outputs the transformed property tensor
                Alternatively use one of the following standard_positions:
                "add", "subtract",
            values (1D iterable):
            states (list of State objects): States in which action will be
                executed. If None, will be executed on all non zero states
            name (String): Name of action, used for data export and readability
        """
        self.states = states
        self.operation = operation
        self.values = torch.HalfTensor(values)
        self.object_property = object_property
        self.name = name

        # define variables which will be used in simulation
        self.value_array = torch.HalfTensor([])


class State():

    def __init__(self, initial_condition, name=""):
        """

        Args:
            name (str): name of state, used for data export and readability
            initial_condition (int): Number of objects in that state
        """
        self.initial_condition = initial_condition
        self.name = name

class ObjectRemovalCondition():
    """
    Define conditions to remove objects from simulations - e.g. if they move
    out of the region of interest. In contrast, min and max values of properties
    prevent objects from moving out.
    """

    def __init__(self, object_properties, operation, threshold):
        """

        Args:
            object_properties:
            operation:
            threshold:
        """
        self.object_properties = object_properties
        self.operation = operation
        self.threshold = threshold

    def operation_sum_smaller_than(self):


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
        if (rates is None) & (lifetimes = None):
            return ValueError("Rates or lifetimes need to be specified for "
                              "the state transition from state"
                              f"{start_state} to state {end_state}.")
        self.start_state = start_state
        self.end_state = end_state
        lifetime_to_rates_factor = torch.log(torch.FloatTensor([2]))
        if rates is None:
            self.rates = lifetime_to_rates_factor / lifetimes
        else:
            self.rates = rates
        self.values = torch.HalfTensor(rates)
        self.time_dependency = time_dependency
        self.name = name

        # initialize variable that will be filled during simulation
        self.value_array = torch.HalfTensor([])
        self.simulation_mask = torch.HalfTensor([])
        self.transition_positions = torch.HalfTensor([])