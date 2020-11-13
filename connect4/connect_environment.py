from rfl.env.abstract_environment import Action, State
from rfl.env.abstract_2player_environment import Abstract2PlayerEnvironment

from typing import Tuple, ClassVar, List

import numpy as np


class ConnectEnvironment(Abstract2PlayerEnvironment):

    action_space: ClassVar[Tuple[Action]] = tuple(range(7))
    directions: ClassVar[Tuple[Tuple[int, int]]] = ((1, 0), (0, 1), (1, 1), (1, -1))

    def __init__(self, player: int):
        super(ConnectEnvironment, self).__init__(np.zeros((2, 7, 6), dtype=np.int), player)
        self.reset()

    def get_state_with_action(self, state: State, action: Action) -> State:
        tmp, self._state = self._state, state
        self._push_action_(action)
        self._state = tmp
        return state

    def get_flipped_state_copy(self) -> State:
        return self._state.copy()[::-1, :, :]

    def get_flipped_state_with_action(self, state: State, action: Action) -> State:
        tmp, self._state = self._state, state
        self._push_action_(action)
        self._state = tmp
        return state[::-1, :, :]

    def get_possible_actions(self) -> List[Action]:
        return [x for x in ConnectEnvironment.action_space if self._state[0, x, -1] == 0 and self._state[1, x, -1] == 0]

    def __top__(self, x: int) -> int:
        for y in range(6):
            if self._state[0, x, y] == 0 and self._state[1, x, y] == 0:
                return y
        return -1

    def __count_dir__(self, x: int, y: int, player: int, vx: int, vy: int) -> int:
        count = 1
        for i in range(1, 4):
            nx, ny = x + i * vx, y + i * vy
            if nx >= 0 and nx < 7 and ny >= 0 and ny < 6 and self._state[player, nx, ny, ] == 1:
                count += 1
            else:
                break

        for i in range(1, 4):
            nx, ny = x - i * vx, y - i * vy
            if nx >= 0 and nx < 7 and ny >= 0 and ny < 6 and self._state[player, nx, ny] == 1:
                count += 1
            else:
                break

        return count

    def _push_action_(self, action: Action):
        if self.is_closed():
            raise Exception("Fatal error: game is already closed !")
        turn = self.turn
        y = self.__top__(action)
        if y == -1:
            self.winner = 1 - turn
            print("Illegal move: making Player ", turn, "lose")
            return
        self._state[turn, action, y] = 1

    def _check_is_closed_from_action_(self, action: Action):
        if self.is_closed():
            return
        y = self.__top__(action)
        if y == -1:
            y = self._state.shape[2] - 1
        else:
            y -= 1
        for (vx, vy) in ConnectEnvironment.directions:
            if self.__count_dir__(action, y, self.turn, vx, vy) >= 4:
                self.winner = self.turn
                return

        if np.sum(self._state) == 42:
            self.winner = 9999


def game_to_string(state: State, player: int) -> str:
    s = "| " + " | ".join([str(x) for x in range(state.shape[1])]) + " |\n"
    s += "-" * (4 + (3 * (state.shape[1] - 1)) + state.shape[1]) + "\n"
    for y in reversed(range(state.shape[2])):
        s += "| "
        for x in range(state.shape[1]):
            if state[player, x, y] == 1:
                s += "X | "
            elif state[1 - player, x, y] == 1:
                s += "O | "
            else:
                s += "  | "
        s += "\n"
    return s


def human_player(env: ConnectEnvironment, player: int) -> Action:
    state = env.get_state_copy()
    print(game_to_string(state, player))
    chosen = -1
    legal = env.get_possible_actions()
    while chosen not in legal:
        chosen = int(input("Action ?"))
    return chosen
