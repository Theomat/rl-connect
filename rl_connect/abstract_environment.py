from abc import ABC, abstractmethod
from typing import List, Callable, ClassVar, TypeVar, Tuple, Iterable

import numpy as np

EnvType = TypeVar('EnvType')


class AbstractEnvironment(ABC):
    action_space: ClassVar[Tuple[int]] = ()

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

    @abstractmethod
    def is_closed(self) -> bool:
        """
        Return whether this environment is closed/terminated or not.
        """
        pass

    def do_episodes(self, policy: Callable[[EnvType], int], n: int = 1) -> List[List[List[np.ndarray, float, np.ndarray]]]:
        """
        Do one episode using the specified policy.

        Parameter
        -----------
        - **policy**: the policy to choose the action
        - **n**: the number of episodes to run

        Return
        -----------
        A list of episodes, an episode is the list of (state, reward, new_state) from this episode.
        """
        episodes = []
        for i in range(n):
            episode = []
            self.environment.reset()
            while not self.environment.is_closed():
                state = self.environment.get_state_copy()
                action = policy(state)
                reward = self.environment.do_action(action)
                episode.append([state, reward, self.environment.get_state_copy()])
            episodes.append(episode)

        return episodes
