from rfl.env.abstract_environment import AbstractEnvironment, Action, State
from rfl.learner.abstract_model_learner import AbstractModelLearner


from abc import ABC, abstractmethod


import numpy as np


class AbstractPolicyModelLearner(AbstractModelLearner, ABC):

    def action_probabilities_for(self, state: State) -> np.ndarray:
        return self.model.predict(np.expand_dims(state, axis=0))

    @property
    def greedy_model_policy(self) -> Policy:
        def policy(env: AbstractEnvironment) -> Action:
            s = env.get_state_copy()
            legal_actions = env.get_possible_actions()
            chosen = np.argmax(self.action_probabilities_for(s)))
            return chosen
        return policy
