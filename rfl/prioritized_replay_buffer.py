from rfl.env.abstract_environment import Episode
from rfl.abstract_replay_buffer import AbstractReplayBuffer, SARSTuple
from rfl.datastructure.ref_counted_list import RefCountedList
from rfl.datastructure.sorted_list import SortedList

from typing import List, TypeVar, Literal, Optional

import numpy as np

Method = TypeVar("Method", Literal["proportional"], Literal["rank"])


class PrioritizedReplayBuffer(AbstractReplayBuffer):

    def __init__(self, size: int = 10000, method: Method = "proportional",
                 alpha: Optional[float] = None, beta: Optional[float] = None,
                 seed: int = 0):
        self._size: int = size
        self._method = method
        self.alpha = alpha or (.7 if method == "rank" else .6)
        self.beta = beta or (.5 if method == "rank" else .4)
        self.epsilon = 10**-3
        self.generator: np.random.Generator = np.random.default_rng(seed)

        self._memory: SortedList = SortedList(key=lambda x: x[0])
        self._episodes: RefCountedList = RefCountedList()

        # A memory is
        # (error, episode_uid, memory_index_in_ep, transition)

    def store(self, episodes: List[Episode]):
        max_error = (self._memory or ((0,),))[0][0]
        for episode in episodes:
            uid = self._episodes.append(episode, len(episode))
            T = len(episode) - 1
            for j, t in enumerate(reversed(episode)):
                self._memory.append((max_error, uid, T - j, t))

        if len(self._memory) > self._size:
            for (uid, _, _, _, _) in self._memory[self._size:]:
                self._episodes.decrease_refs(uid, 1)
            self._memory = self._memory[:self._size]

    def sample(self, size: int, nsteps: int) -> List[SARSTuple]:
        if self._method == "rank":
            probabilities = np.asarray([1 / (i + 1) for i in range(len(self._memory))])
        else:
            probabilities = np.asarray([(-e + self.epsilon) for (e, _, _, _) in self._memory])
        probabilities = np.power(probabilities, self.alpha, out=probabilities)
        probabilities /= np.sum(probabilities)
        memories = self.generator.choice(np.arange(0, len(self._memory)), size, p=probabilities, shuffle=False)
        self._need_updates = memories
        # Now retrieve
        output = []
        weights = []
        probabilities *= size
        weights = np.power(probabilities, -self.beta, out=probabilities)
        weights /= np.max(weights)
        for g_index in memories:
            (_, episode_uid, memory_index, t) = self._memory[g_index]
            (state, action, reward) = t
            afterwards = []
            episode = self._episodes[episode_uid]

            i = memory_index
            for j in range(1, nsteps + 1):
                if i + j < len(episode):
                    afterwards.append(episode[i + j])
            output.append((state, action, reward, afterwards, weights[g_index]))
        return output

    def step(self, losses: np.ndarray, beta: float):
        for i in range(losses.shape[0]):
            index = self._need_updates[i]
            (_, uid, i, t) = self._memory[index]
            self._memory.replace(index, (-losses[i], uid, i, t))
        self.beta = beta
