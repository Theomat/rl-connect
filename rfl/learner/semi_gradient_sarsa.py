from rfl.learner.abstract_state_model_learner import AbstractStateModelLearner
from rfl.abstract_replay_buffer import SARSTuple


import torch

from typing import List, Tuple, Optional

import numpy as np


class SemiGradientSARSALearner(AbstractStateModelLearner):
    def configure(self,  gamma: float = 1, steps: int = 1, **kwargs):
        self.gamma = gamma
        self.steps = steps

    def train(self, **kwargs):
        transitions: List[SARSTuple] = self.replay_buffer.sample(self.batch_size, self.steps)
        X, Y, W = [], [], []
        for trans in transitions:
            x, y, w = self._transition_to_dataset_(trans)
            X.append(x)
            Y.append(y)
            if w:
                W.append(w)

        # Dataset is now ready
        X = np.asarray(X, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.float32)
        X = torch.FloatTensor(X).to(self.device)
        y_true = torch.FloatTensor(Y).to(self.device)
        weights = None
        if W:
            weights = torch.FloatTensor(np.asarray(W, dtype=np.float32)).to(self.device)
        # Actual learning
        self.model.zero_grad()
        y_pred = self.model(X)
        if W:
            loss = self.loss_fn(y_pred.flatten(), y_true, reduction='none')
            ef_loss = torch.mean(loss * weights)
        else:
            loss = self.loss_fn(y_pred.flatten(), y_true)
            ef_loss = loss
        self.optimizer.zero_grad()
        ef_loss.backward()
        self.optimizer.step()
        nloss = loss.detach().numpy()
        self.metrics["training.loss"].append(np.mean(nloss))
        return nloss

    def _transition_to_dataset_(self, transition: SARSTuple) -> Tuple[np.ndarray, float, Optional[float]]:
        (state, action, reward, afterwards, w) = transition
        T: int = len(afterwards)
        G = reward + np.sum([self.gamma**(i + 1) * r for i, (s, a, r) in enumerate(afterwards[:-1])])
        if T == self.steps:
            last_state = afterwards[-1][0].copy()
            G += self.gamma**self.steps * self.value_of_state(last_state).detach().numpy()
        return state, G, w
