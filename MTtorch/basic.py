import torch

class ObjectProperty():

    def __init__(self, min_value=0, max_value=1, start_value=None,
                 initial_condition=None, name=""):
        """
        Args:
            min_value (float): Minimum allowed value
            max_value (float): Maximum allowed value
            start_value (float): Value at which new objects will be initialized
            initial_condition (float, func): define the values used at
                the  beginning of each simulation. If None and start_value None,
                it will be assigned a random values from min_value to max_value.
                If function, should take the number of events and output
                1D array with corresponding values.
            name (String): Name of property, used for data export
                and readibility
        """
        self.min_value = min_value
        self.max_value = max_value
        self.start_value = start_value
        self.initial_condition = initial_condition
        self.name = name

        # initialize variables used in simulation
        self.value_array = torch.HalfTensor([])


class Action():

    def __init__(self, object_property, operation, values, states = None,
                 name=""):
        """

        Args:
            object_property (object_property object):
            operation (func): function that takes the property tensor,
                a tensor of the values (same shape as the property tensor) and
                the time tensor and then outputs the transformed property tensor
                the time tensor and then outputs the transformed property tensor
                Alternatively use one of the following standard_positions:
                "add", "subtract",
            values (1D iterable):
            states (list of state objects): States in which action will be
                executed. If None, will be executed on all non zero states
            name (String): Name of action, used for data export and readibility
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
            name (str): name of state, used for data export and readibility
            initial_condition (int): Number of objects in that state
        """
        self.initial_condition = initial_condition
        self.name = name

class StateTransition():

    def __init__(self, start_state=None, end_state=None, rates=None, lifetimes=None,
                 time_dependency = None, name=""):
        """

        Args:
            start_state (state object): Name of state at which the transition
                starts, If None, then the end_state is created (from state 0)
            end_state (state object): Name of state at which the transition ends
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
            name (str): Name of transition, used for data export and readibility
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