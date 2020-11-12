from rfl.abstract_environment import AbstractEnvironment, Episode
from rfl.abstract_replay_buffer import AbstractReplayBuffer, SARSTuple
from rfl.policies import Policy

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional

from tensorflow.keras import Model
from tensorflow.keras.callbacks import History

import numpy as np


class AbstractModelLearner(ABC):
    def __init__(self, env: AbstractEnvironment, model: Model, replay_buffer: AbstractReplayBuffer):
        self.env: AbstractEnvironment = env
        self.model: Model = model
        self.replay_buffer: AbstractReplayBuffer = replay_buffer
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
        self.replay_buffer.store(episodes)

    @abstractmethod
    def _transitions_to_dataset_(self, transitions: List[SARSTuple]) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Transform a list of transitions into a dataset.

        Parameters
        -----------
        - **transitions**: the transitions to be converted

        Return
        -----------
        The tuple (X, y, w) which is the training dataset.
        w may be None and is the weight of each example.
        """
        pass

    def train(self, sample_size: int, **kwargs) -> History:
        transitions: List[SARSTuple] = self.replay_buffer.sample(sample_size)
        x, y, w = self._transitions_to_dataset_(transitions)
        return self.model.fit(x, y, sample_weight=w, **kwargs)

    def produce_metrics(self, episode: Episode):
        self.metrics["episode.reward"].append(sum([r for (state, action, r) in episode]))
        self.metrics["episode.length"].append(len(episode))

    @abstractmethod
    @property
    def greedy_model_policy(self) -> Policy:
        """
        Return the greedy policy following this model.
        """
        pass
