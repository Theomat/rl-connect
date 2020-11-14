from rfl.env.abstract_environment import AbstractEnvironment, Episode
from rfl.abstract_replay_buffer import AbstractReplayBuffer
from rfl.policies import Policy

from typing import List

from abc import ABC, abstractmethod


class AbstractModelLearner(ABC):
    def __init__(self, env: AbstractEnvironment, model, replay_buffer: AbstractReplayBuffer):
        self.env: AbstractEnvironment = env
        self.model = model
        self.replay_buffer: AbstractReplayBuffer = replay_buffer
        self.metrics: dict[str, float] = {
            "episode.reward": [],
            "episode.length": [],
            "training.loss": []
        }

    def setup_training(self, loss_fn, optimizer, batch_size: int = 32, device: str = 'cpu', **kwargs):
        self.batch_size = batch_size
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device

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
            self.produce_metrics(episode)
        self.replay_buffer.store(episodes)

    @abstractmethod
    def train(self, **kwargs: dict):
        pass

    def produce_metrics(self, episode: Episode):
        self.metrics["episode.reward"].append(sum([r for (state, action, r) in episode]))
        self.metrics["episode.length"].append(len(episode))

    @property
    @abstractmethod
    def greedy_model_policy(self) -> Policy:
        """
        Return the greedy policy following this model.
        """
        pass
