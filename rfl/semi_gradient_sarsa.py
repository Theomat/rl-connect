from rfl.abstract_model_learner import AbstractModelLearner
from rfl.abstract_environment import Episode


from typing import List, Tuple

import numpy as np


class SemiGradientSARSALearner(AbstractModelLearner):
    def configure(self, gamma: float, steps: int = 1, **kwargs):
        self.gamma = gamma
        self.steps = steps

    def episode_to_dataset(self, episode: Episode) -> List[Tuple[np.ndarray, np.ndarray]]:
        dataset = []
        T = len(episode) - 1
        n = self.steps
        tau = 0
        t = 0
        rewards = [r for (s, a, r) in episode]
        values = self.value_of_states([state for (state, a, r) in episode[n:]])
        while tau != T - 1:
            tau = t - n + 1
            if tau >= 0:
                G = np.sum([self.gamma**(i-tau-1) * rewards[i] for i in range(tau + 1, min(tau + n, T) + 1)])
                if tau + n < T:
                    G += self.gamma**n * values[t]
                dataset.append((episode[tau][0], G))
            t += 1
        return dataset
