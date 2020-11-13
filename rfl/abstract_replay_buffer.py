from rfl.env.abstract_environment import Episode, Action, State, Transition


from abc import ABC, abstractmethod
from typing import List, Tuple, Optional

SARSTuple = Tuple[State, Action, float, List[Transition], Optional[float]]


class AbstractReplayBuffer(ABC):
    @abstractmethod
    def sample(self, size: int, nsteps: int, **kwargs: dict) -> List[SARSTuple]:
        """
        Sample the specified number of transitions from this buffer.

        Parameters
        -----------
        - **size**: the number of transitions to be sampled
        - **nsteps**: the number of future steps to get

        Return
        -----------
        The list of transitions.
        """
        pass

    @abstractmethod
    def store(self, episodes: List[Episode]):
        """
        Store the specified episodes into this replay buffer.

        Parameters
        -----------
        - **episodes**: the list of episodes to be stored
        """
        pass
