from rfl.abstract_environment import AbstractEnvironment, Action, State
from rfl.abstract_2player_environment import Abstract2PlayerEnvironment

from typing import Callable

import numpy as np

Policy = Callable[[AbstractEnvironment], Action]


def epsilon_greedy(epsilon: float, greedy_policy: Policy, seed=None) -> Policy:
    """
    Transform a greedy policy into an espilon greedy policy.

    Parameters
    -----------
    - **epsilon**: the chance of taking a random action
    - **greedy_policy**: the greedy policy to follow
    - **seed**: the seed to be passed to ```numpy.random.default_rng```

    Return
    -----------
    The new espilon greedy policy
    """
    generator = np.random.default_rng(seed)

    def f(env: AbstractEnvironment) -> Action:
        if generator.uniform() <= epsilon:
            # Random
            actions = list(env.get_possible_actions())
            return generator.choice(actions)
        else:
            return greedy_policy(env)

    return f


def random_policy(env: AbstractEnvironment) -> Action:
    """
    The random policy, it uses the global PRNG of numpy.
    It takes a random legal actions.
    """
    actions = list(env.get_possible_actions())
    return np.random.choice(actions)


def simple_policy(env: Abstract2PlayerEnvironment, player: int) -> Action:
    """
    A policy that, if able to win by doing a legal action will so, otherwise it will play randomly.
    Currently restricted to 2 player games.
    """
    actions = list(env.get_possible_actions())
    for action in actions:
        env.push()
        env.do_action(action)
        if env.is_closed() and env.winner == player:
            return action
        env.pop()
    return np.random.choice(actions)


def greedy_action_values(action_values: Callable[[State], np.ndarray]) -> Policy:
    """
    Create a greedy policy from an action values function.

    Parameters
    -----------
    - **action_values**: a fucntion that maps a state to the values of the different actions

    Return
    -----------
    The new greedy action values policy
    """
    def f(env: AbstractEnvironment) -> Action:
        values = action_values(env.get_state_copy())
        return np.argmax(values)
    return f
