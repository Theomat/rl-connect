from rfl.abstract_environment import Episode
from rfl.abstract_replay_buffer import AbstractReplayBuffer, SARSTuple

from typing import List

import numpy as np


class UniformReplayBuffer(AbstractReplayBuffer):

    def __init__(self, size: int = 10000, seed: int = 0):
        self._size: int = size
        self._memory: List = []
        self.generator: np.random.Generator = np.random.default_rng(seed)
        self._episodes: List[Episode] = []

    def store(self, episodes: List[Episode]):
        i = len(self._episodes)
        for episode in episodes:
            T = len(episode) - 1
            for j, (state, action, reward) in enumerate(reversed(episode)):
                self._memory.append((i, T - j, state, action, reward))
            self._episodes.append(episode)
            i += 1
        if len(self._memory) > self._size:
            self._memory = self._memory[-self._size:]

    def sample(self, size: int, nsteps: int) -> List[SARSTuple]:
        memories = self.generator.integers(0, len(self._memory), size, dtype=np.int)
        output = []
        for g_index in memories:
            (episode_index, memory_index, state, action, reward) = self._memory[g_index]
            afterwards = []
            episode = self._episodes[episode_index]
            i = memory_index
            for j in range(1, nsteps + 1):
                if i + j < len(episode):
                    afterwards.append(episode[i + j])
            output.append((state, action, reward, afterwards, None))
        return output
