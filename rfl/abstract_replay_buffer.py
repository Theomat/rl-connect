from rfl.abstract_environment import Episode, Action, State


from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, TypeVar

SARSTuple = TypeVar('SARSTuple',
                    Tuple[State, Action, float, Optional[State]],
                    Tuple[State, Action, float, Optional[State], float])


class AbstractReplayBuffer(ABC):
    @abstractmethod
    def sample(self, size: int, **kwargs: dict) -> List[SARSTuple]:
        """
        Sample the specified number of transitions from this buffer.

        Parameters
        -----------
        - **size**: the number of transitions to be sampled

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
