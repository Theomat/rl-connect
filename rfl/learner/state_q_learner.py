from rfl.learner.semi_gradient_sarsa import SemiGradientSARSALearner
from rfl.abstract_replay_buffer import SARSTuple
from rfl.env.abstract_environment import State

from typing import Tuple, Optional

import numpy as np


class StateQLearner(SemiGradientSARSALearner):

    def __best_value(self, state: State):
        self.env.push()
        self.env.set_state(state)
        values = self.value_of_state_with_actions(state, self.env.get_possible_actions()).detach().numpy()
        self.env.pop()
        return np.max(values)

    def _transition_to_dataset_(self, transition: SARSTuple) -> Tuple[np.ndarray, float, Optional[float]]:
        (state, action, reward, afterwards, w) = transition
        T: int = len(afterwards)
        G = reward + np.sum([self.gamma**(i + 1) * r for i, (s, a, r) in enumerate(afterwards[:-1])])
        if T == self.steps:
            last_state = afterwards[-1][0].copy()
            value = self.__best_value(last_state)
            G += self.gamma**self.steps * value
        return state, G, w
