from rfl.abstract_environment import AbstractEnvironment, Episode, Action, State
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

    def value_of_state(self, state: State) -> float:
        return self.model.predict(np.expand_dims(state, axis=0))

    def value_of_states(self, states: List[State]) -> np.ndarray:
        if len(states) > 0:
            return self.model.predict(np.asarray(states, dtype=np.float32))
        else:
            return []

    def value_of_state_action(self, state: State, action: Action) -> float:
        state = self.env.get_state_with_action(state.copy(), action)
        return self.model.predict(np.expand_dims(state, axis=0))

    def value_of_state_actions(self, state: State, actions: List[Action]) -> np.ndarray:
        return self.value_of_states([self.env.get_state_with_action(state.copy(), action) for action in actions])

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
