from rfl.env.abstract_environment import Episode
from rfl.abstract_replay_buffer import AbstractReplayBuffer, SARSTuple

from typing import List, TypeVar, Literal, Optional

import numpy as np

Method = TypeVar("Method", Literal["proportional"], Literal["rank"])


class PrioritizedReplayBuffer(AbstractReplayBuffer):

    def __init__(self, size: int = 10000, method: Method = "proportional",
                 alpha: Optional[float] = None, beta: Optional[float] = None,
                 seed: int = 0):
        self._size: int = size
        self._memory: List = []
        self._errors: List = []
        self._method = method
        self.alpha = alpha or (.7 if method == "rank" else .6)
        self.beta = beta or (.5 if method == "rank" else .4)
        self.epsilon = 10**-3
        self.generator: np.random.Generator = np.random.default_rng(seed)

        self._episodes: List[Episode] = []

        # A memory is
        # (episode_index, memory_index_in_ep, transition, priority)

    def store(self, episodes: List[Episode]):
        i = len(self._episodes)
        max_error = np.max(self._errors) if self._errors else 1
        for episode in episodes:
            T = len(episode) - 1
            for j, t in enumerate(reversed(episode)):
                self._memory.append((i, T - j, t))
                self._errors.append(max_error)
            self._episodes.append(episode)
            i += 1
        if len(self._memory) > self._size:
            self._memory = self._memory[-self._size:]
            self._errors = self._errors[-self._size:]

    def sample(self, size: int, nsteps: int) -> List[SARSTuple]:
        if self._method == "rank":
            raise Exception("Not yet implemented")
            probabilities = np.asarray([])
        else:
            probabilities = np.asarray([(e + self.epsilon) for e in self._errors])
        probabilities = np.power(probabilities, self.alpha, out=probabilities)
        probabilities /= np.sum(probabilities)
        memories = self.generator.choice(np.arange(0, len(self._memory)), size, p=probabilities, shuffle=False)
        self._need_updates = memories
        # Now retrieve
        output = []
        weights = []
        for g_index in memories:
            (episode_index, memory_index, t) = self._memory[g_index]
            (state, action, reward) = t
            weight = (1 / size * 1 / probabilities[g_index]) ** self.beta
            weights.append(weight)
            afterwards = []
            episode = self._episodes[episode_index]
            i = memory_index
            for j in range(1, nsteps + 1):
                if i + j < len(episode):
                    afterwards.append(episode[i + j])
            output.append((state, action, reward, afterwards))
        mw = np.max(weights)
        return [(s, a, r, aft, w / mw) for (s, a, r, aft), w in zip(output, weights)]

    def step(self, losses: np.ndarray, beta: float):
        for i in range(losses.shape[0]):
            index = self._need_updates[i]
            self._errors[index] = losses[i]
        self.beta = beta
