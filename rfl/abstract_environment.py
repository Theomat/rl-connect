from abc import ABC, abstractmethod
from typing import List, Callable, ClassVar, TypeVar, Tuple, Iterable

import numpy as np

from tqdm.auto import trange

EnvType = TypeVar('EnvType')
State = np.ndarray
Action = int
Transition = Tuple[State, Action, float]
Episode = List[Transition]


class AbstractEnvironment(ABC):
    action_space: ClassVar[Tuple[Action]] = ()

    @abstractmethod
    def reset(self):
        """
        Reset the state of this environement to default.
        """
        pass

    @abstractmethod
    def get_possible_actions(self) -> Iterable[Action]:
        """
        Get the list of possible actions from the current state of the environment.
        """
        pass

    @abstractmethod
    def get_state_copy(self) -> State:
        """
        Get a state copy of this environment as a numpy array.
        """
        pass

    @abstractmethod
    def get_state_with_action(self, state: State, action: Action) -> State:
        """
        Get a state copy of this environment with the specified action taken as a numpy array.
        """
        pass

    @abstractmethod
    def do_action(self, action: Action) -> float:
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

    def do_episodes(self, policy: Callable[[EnvType], Action], n: int = 1) -> List[Episode]:
        """
        Do one episode using the specified policy.

        Parameter
        -----------
        - **policy**: the policy to choose the action
        - **n**: the number of episodes to run

        Return
        -----------
        A list of episodes, an episode is the list of (state, action, reward) from this episode.
        """
        episodes = []
        for i in trange(n, desc="episode"):
            episode = []
            self.reset()
            while not self.is_closed():
                state = self.get_state_copy()
                action = policy(self)
                reward = self.do_action(action)
                episode.append([state, action, reward])
            episodes.append(episode)

        return episodes
