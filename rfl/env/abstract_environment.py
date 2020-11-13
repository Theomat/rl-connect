from abc import ABC, abstractmethod
from typing import List, Callable, ClassVar, TypeVar, Tuple, Iterable

import numpy as np

EnvType = TypeVar('EnvType')
State = np.ndarray
Action = int
Transition = Tuple[State, Action, float]
Episode = List[Transition]


class AbstractEnvironment(ABC):
    action_space: ClassVar[Tuple[Action]] = ()

    def __init__(self, initial_state):
        self._state = initial_state
        self._saves = []

    def set_state(self, state: State):
        """
        Set the state of the environment to the specified state.

        Parameters
        -----------
        - **state**: the new state of this environment
        """
        self._state = state

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

    def push(self):
        """Save the current state of the environment to restore it later."""
        self._saves.append(self.get_state_copy())

    def pop(self):
        """Restore the current state of the environment to a previous state."""
        self.set_state(self._saves.pop(-1))

    def get_state_copy(self) -> State:
        """
        Get a state copy of this environment as a numpy array.
        """
        return self._state.copy()

    def get_state_with_action(self, state: State, action: Action) -> State:
        """
        Get a state copy of this environment with the specified action taken as a numpy array.
        """
        self.push()
        self.set_state(state)
        self.do_action(action)
        output = self._state
        self.pop()
        return output

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
        for i in range(n):
            episode = []
            self.reset()
            while not self.is_closed():
                state = self.get_state_copy()
                action = policy(self)
                reward = self.do_action(action)
                episode.append([state, action, reward])
            episodes.append(episode)

        return episodes
