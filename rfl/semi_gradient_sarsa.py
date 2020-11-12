from rfl.abstract_state_model_learner import AbstractStateModelLearner
from rfl.abstract_replay_buffer import SARSTuple


import torch

from typing import List, Tuple

import numpy as np


class SemiGradientSARSALearner(AbstractStateModelLearner):
    def configure(self, gamma: float, batch_size: int, loss_fn, steps: int = 1, device: str = 'cpu', **kwargs):
        self.gamma = gamma
        self.steps = steps
        self.batch_size = batch_size
        self.loss_fn = loss_fn
        super(SemiGradientSARSALearner, self).configure(device)

    def train(self, **kwargs):
        transitions: List[SARSTuple] = self.replay_buffer.sample(self.batch_size, self.steps)
        X, Y = [], []
        for trans in transitions:
            x, y = self._transition_to_dataset_(trans)
            X.append(x)
            Y.append(y)
        # Dataset is now ready
        X = np.asarray(X, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.float32)
        X = torch.FloatTensor(X).to(self.device)
        y_true = torch.FloatTensor(Y).to(self.device)
        # Actual learning
        self.model.zero_grad()
        y_pred = self.model(X)
        loss = self.loss_fn(y_pred.flatten(), y_true)
        return loss

    def _transition_to_dataset_(self, transition: SARSTuple) -> Tuple[np.ndarray, float]:
        (state, action, reward, afterwards, _) = transition
        T: int = len(afterwards)
        rewards = [r for (s, a, r) in afterwards]
        G = np.sum([self.gamma**(i) * rewards[i] for i in range(min(T, self.steps - 1))])
        if T == self.steps:
            G += self.gamma**self.steps * self.value_of_state(afterwards[-1][0])
        return state, G
