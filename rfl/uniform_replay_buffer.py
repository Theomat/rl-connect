from rfl.abstract_environment import Episode
from rfl.abstract_replay_buffer import AbstractReplayBuffer, SARSTuple

from typing import List

import numyp as np


class UniformReplayBuffer(AbstractReplayBuffer):

    def __init__(self, size: int = 10000, seed: int = 0):
        self._size: int = size
        self._memory: List[SARSTuple] = []
        self.generator: np.random.Generator = np.random.default_rng(seed)

    def store(self, episodes: List[Episode]):
        for episode in episodes:
            next_state = None
            for (state, action, reward) in reversed(episode):
                self._memory.append((state, action, reward, next_state))
                next_state = state

        if len(self._memory) > self._size:
            self._memory = self._memory[-self._size:]

    def sample(self, size: int) -> List[SARSTuple]:
        return self.generator.choice(self._memory, size)
