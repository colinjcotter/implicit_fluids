import abc
from ics import set_initial_conditions()

class BaseModel(object, metaclass=abc.ABC):
    def __init__(self):
        """
        Model base class for implicit fluids.
        """
        self.allocate()
        self.build_eqn()
        set_initial_condition(self)

    @abc.abstractmethod
    def allocate(self):
        """
        Allocate U0 and any Function coefficients
        required in the equation system.
        """
        pass

    def build_eqn(self):
        """
        Construct the equation system.
        """
        pass

    @property
    @abc.abstractmethod
    def U0(self):
        """
        Return the Function describing the model state,
        updated by the timestepper.
        """
        pass

    @property
    @abc.abstractmethod
    def eqn(self):
        """
        Return the UFL form describing the model equation system,
        written in terms of U0.
        """
        pass
