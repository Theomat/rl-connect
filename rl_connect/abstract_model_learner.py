from rl_connect.abstract_environment import AbstractEnvironment, Episode
from rl_connect.policies import Policy, greedy_action_values

from abc import ABC, abstractmethod
from typing import List, Tuple

from tf.keras import Model
from tf.keras.callbaks import History

import numpy as np


class AsbtractModelLearner(ABC):
    def __init__(self, env: AbstractEnvironment, model: Model, memory_size: int):
        self.env: AbstractEnvironment = env
        self.model: Model = model
        self._memory: List[Tuple[np.ndarray, np.ndarray]] = []
        self.memory_size: int = memory_size
        self.metrics: dict[str, float] = {
            "reward": [],
            "episode_length": []
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
        dataset = np.random.choice(self._memory, size=sample_size, replace=False)
        x = dataset[:, 0]
        y = dataset[:, 1]
        return self.model.fit(x, y, **kwargs)

    @property
    def greedy_model_policy(self) -> Policy:
        return greedy_action_values(self.model.predict)

    def produce_metrics(self, episode: Episode):
        self.metrics["episode.reward"].append(sum([r for (state, action, r) in episode]))
        self.metrics["episode.length"].append(len(episode))
