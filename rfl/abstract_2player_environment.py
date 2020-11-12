from rfl.abstract_environment import AbstractEnvironment, Action, State

from abc import ABC, abstractmethod
from typing import Callable


class Abstract2PlayerEnvironment(AbstractEnvironment, ABC):

    def __init__(self, player: int = 0):
        self.player: int = 0
        self.turn: int = 0
        self.other_player: Callable[[State, int], int] = None
        self.winner: int = 0
        self.play_reward: float = 0
        self.win_reward: float = 1

    @abstractmethod
    def get_flipped_state_copy(self) -> State:
        """
        Get a flipped state copy of this environment as a numpy array.
        """
        pass

    @abstractmethod
    def get_flipped_state_with_action(self, state: State, action: Action) -> State:
        """
        Get a flipped state copy of this environment with the specified action taken (before flipping) as a numpy array.
        """
        pass

    def attach_second_player(self, other_player: Callable[[AbstractEnvironment, int], int]) -> None:
        """
        Attach a second player to the environment.

        Parameters
        -----------
        - **other_player**: a function which
        """
        self.other_player = other_player

    def next_turn(self):
        """
        Move to the turn of the next player.
        """
        self.turn = 1 - self.turn

    def play_second_player(self):
        """
        Play one turn of the second player.
        """
        if not self.is_closed() and self.turn != self.player:
            self.do_action(self.other_player(self, self.turn))

    @abstractmethod
    def _push_action_(self, action: Action):
        """
        Modify the internal state to do the specified action.

        Parameters
        -----------
        - **action** the action to be taken
        """
        pass

    @abstractmethod
    def _check_is_closed_from_action_(self, action: Action):
        """
        Called internally after the specified action was taken to update the internal state of the game.

        Parameters
        -----------
        - **action**: the action that was just taken
        """
        pass

    def do_action(self, action: Action) -> float:
        if self.closed:
            raise Exception("Fatal error: game is already closed !")
        self._push_action_(action)

        self._check_is_closed_from_action_(action)
        reward = self.play_reward
        if not self.is_closed():
            self.next_turn()
            self.play_second_player()

        if self.is_closed():
            reward = self.win_reward * (1 if self.winner == self.player else -1)
        return reward
