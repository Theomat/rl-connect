from rfl.env.abstract_environment import Episode
from rfl.abstract_replay_buffer import AbstractReplayBuffer, SARSTuple
from rfl.datastructure.ref_counted_list import RefCountedList

from typing import List

import numpy as np


class UniformReplayBuffer(AbstractReplayBuffer):

    def __init__(self, size: int = 10000, seed: int = 0):
        self._size: int = size
        self._memory: List = []
        self.generator: np.random.Generator = np.random.default_rng(seed)
        self._episodes: RefCountedList = RefCountedList()

    def store(self, episodes: List[Episode]):
        for episode in episodes:
            uid = self._episodes.append(episode, len(episode))
            T = len(episode) - 1
            for j, (state, action, reward) in enumerate(reversed(episode)):
                self._memory.append((uid, T - j, state, action, reward))
            self._episodes.append(episode)
        if len(self._memory) > self._size:
            for (uid, _, _, _, _) in self._memory[:-self._size]:
                self._episodes.decrease_refs(uid, 1)
            self._memory = self._memory[-self._size:]

    def sample(self, size: int, nsteps: int) -> List[SARSTuple]:
        memories = self.generator.integers(0, len(self._memory), size, dtype=np.int)
        output = []
        for g_index in memories:
            (episode_uid, memory_index, state, action, reward) = self._memory[g_index]
            afterwards = []
            episode = self._episodes[episode_uid]
            i = memory_index
            for j in range(1, nsteps + 1):
                if i + j < len(episode):
                    afterwards.append(episode[i + j])
            output.append((state, action, reward, afterwards, None))
        return output
