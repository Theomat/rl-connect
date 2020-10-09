from typing import Tuple, Iterable
from abc import ABC, abstractmethod

import numpy as np


class AbstractEnvironment(ABC):
    action_space: Tuple = ()

    @abstractmethod
    def reset(self):
        """
        Reset the state of this environement to default.
        """
        pass

    @abstractmethod
    def get_possible_actions(self) -> Iterable[int]:
        """
        Get the list of possible actions from the current state of the environment.
        """
        pass

    @abstractmethod
    def get_state_copy(self) -> np.ndarray:
        """
        Get a state copy of this environment as a numpy array.
        """
        pass

    @abstractmethod
    def do_action(self, action: int) -> float:
        """
        Do the specified action in this environment and return the reward from doing this action.
        """
        pass
