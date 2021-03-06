from rfl.env.abstract_environment import AbstractEnvironment, Action, State
from rfl.learner.abstract_model_learner import AbstractModelLearner
from rfl.policies import Policy

from abc import ABC
from typing import List

import torch
import numpy as np


class AbstractStateModelLearner(AbstractModelLearner, ABC):

    def value_of_state(self, state: State) -> float:
        return self.model(torch.FloatTensor(np.expand_dims(state, axis=0)).to(self.device))

    def value_of_states(self, states: List[State]) -> np.ndarray:
        if len(states) > 0:
            return self.model(torch.FloatTensor(np.asarray(states, dtype=np.float32)).to(self.device))
        else:
            return []

    def value_of_state_action(self, state: State, action: Action) -> float:
        state = self.env.get_state_with_action(state.copy(), action)
        return self.model(torch.FloatTensor(np.expand_dims(state, axis=0)).to(self.device))

    def value_of_state_actions(self, state: State, actions: List[Action]) -> np.ndarray:
        return self.value_of_states([self.env.get_state_with_action(state.copy(), action) for action in actions])

    @property
    def greedy_model_policy(self) -> Policy:
        def policy(env: AbstractEnvironment) -> Action:
            s = env.get_state_copy()
            legal_actions = env.get_possible_actions()
            return legal_actions[np.argmax(self.value_of_state_actions(s, legal_actions).detach().numpy())]
        return policy
