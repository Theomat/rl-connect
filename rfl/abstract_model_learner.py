from rfl.abstract_environment import AbstractEnvironment, Episode, Action, State
from rfl.policies import Policy

from abc import ABC, abstractmethod
from typing import List, Tuple

from tensorflow.keras import Model
from tensorflow.keras.callbacks import History

import numpy as np


class AbstractModelLearner(ABC):
    def __init__(self, env: AbstractEnvironment, model: Model, memory_size: int):
        self.env: AbstractEnvironment = env
        self.model: Model = model
        self._memory: List[Tuple[np.ndarray, np.ndarray]] = []
        self.memory_size: int = memory_size
        self.metrics: dict[str, float] = {
            "episode.reward": [],
            "episode.length": []
        }

    @abstractmethod
    def configure(self, **kwargs):
        """
        Method to call to the configure the model learner.
        """
        pass

    def produce_episodes(self, policy: Policy, episodes: int) -> None:
        """
        Produce the specified number of episodes, process them and add them to the data set.

        Parameters
        -----------
        - **policy**: the policy to be used  to produce the episodes
        - **episodes**: the number of episodes to produce
        """
        episodes: List[Episode] = self.env.do_episodes(policy, n=episodes)
        for episode in episodes:
            for x, y in self.episode_to_dataset(episode):
                self._memory.append((x, y))
            self.produce_metrics(episode)
        # Trim if memory is too big
        if len(self._memory) > self.memory_size:
            self._memory = self._memory[-self.memory_size:]

    @abstractmethod
    def episode_to_dataset(self, episode: Episode) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Transform an episode into a list of data points X, Y.

        Parameters
        -----------
        - **episode**: the episode to be converted

        Return
        -----------
        The list of tuples (X, y) that the model should learn.
        """
        pass

    def train(self, sample_size: int, **kwargs) -> History:
        if sample_size >= len(self._memory):
            x = np.asarray([x for (x, y) in self._memory], dtype=np.float32)
            y = np.asarray([y for (x, y) in self._memory], dtype=np.float32)
        else:
            indices = np.random.choice(len(self._memory), size=sample_size, replace=False)
            x = np.asarray([x for (x, y) in self._memory], dtype=np.float32)[indices]
            y = np.asarray([y for (x, y) in self._memory], dtype=np.float32)[indices]
        return self.model.fit(x, y, **kwargs)

    def value_of_state(self, state: State) -> float:
        return self.model.predict(np.expand_dims(state, axis=0))

    def value_of_states(self, states: List[State]) -> np.ndarray:
        return self.model.predict(np.asarray(states, dtype=np.float32))

    def value_of_state_action(self, state: State, action: Action) -> float:
        state = self.env.get_state_with_action(state, action)
        return self.model.predict(np.expand_dims(state, axis=0))

    def value_of_state_actions(self, state: State, actions: List[Action]) -> np.ndarray:
        states = np.asarray([self.env.get_state_with_action(state, action) for action in actions], dtype=np.float32)
        return self.model.predict(states)

    @property
    def greedy_model_policy(self) -> Policy:
        def policy(env: AbstractEnvironment) -> Action:
            s = env.get_state_copy()
            legal_actions = env.get_possible_actions()
            return legal_actions[np.argmax(self.value_of_state_actions(s, legal_actions))]
        return policy

    def produce_metrics(self, episode: Episode):
        self.metrics["episode.reward"].append(sum([r for (state, action, r) in episode]))
        self.metrics["episode.length"].append(len(episode))
