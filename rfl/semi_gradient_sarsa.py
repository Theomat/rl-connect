from rfl.abstract_model_learner import AbstractModelLearner
from rfl.abstract_environment import Episode


from typing import List, Tuple

import numpy as np


class SemiGradientSARSALearner(AbstractModelLearner):
    def configure(self, gamma, **kwargs):
        self.gamma = gamma

    def episode_to_dataset(self, episode: Episode) -> List[Tuple[np.ndarray, np.ndarray]]:
        dataset = []
        values = self.value_of_states([state for (state, a, r) in episode[1:]])
        for i, (state, action, reward) in enumerate(reversed(episode)):
            if i == 0:
                dataset.append((state, reward))
            else:
                dataset.append((state, reward + self.gamma * values[-i]))

        return dataset
