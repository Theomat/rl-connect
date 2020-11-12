from rfl.abstract_environment import AbstractEnvironment, Action, State

from abc import ABC, abstractmethod
from typing import Callable


class Abstract2PlayerEnvironment(AbstractEnvironment, ABC):

    def __init__(self, initial_state: State, player: int = 0):
        super(Abstract2PlayerEnvironment, self).__init__(initial_state)
        self._initial_board = initial_state.copy()
        self.player: int = 0
        self.turn: int = 0
        self.other_player: Callable[[State, int], int] = None
        self.winner: int = -1
        self.play_reward: float = 0
        self.win_reward: float = 1
        self.draw_reward: float = 0

    def reset(self):
        self.set_state(self._initial_board.copy())
        self.turn = 0
        self.winner = -1

    @abstractmethod
    def get_flipped_state_copy(self) -> State:
        """
        Get a flipped state copy of this environment as a numpy array.
        """
        pass

    @abstractmethod
    def get_flipped_state_with_action(self, state: State, action: Action) -> State:
        """
        Get a flipped state with the specified action taken (before flipping) as a numpy array.
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
        if self.is_closed():
            raise Exception("Fatal error: game is already closed !")
        self._push_action_(action)

        self._check_is_closed_from_action_(action)
        reward = self.play_reward
        if not self.is_closed():
            self.next_turn()
            self.play_second_player()

        if self.is_closed():
            if self.winner == self.player:
                reward = self.win_reward
            elif abs(self.winner - self.player) == 1:
                reward = -self.win_reward
            else:
                reward = self.draw_reward
        return reward

    def is_closed(self) -> bool:
        return self.winner >= 0

    def push(self):
        self._saves.append([self.get_state_copy(), self.turn, self.winner])

    def pop(self):
        state, self.turn, self.winner = self._saves.pop(-1)
        self.set_state(state)
