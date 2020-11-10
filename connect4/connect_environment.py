from rfl.abstract_environment import Action, State
from rfl.abstract_2player_environment import Abstract2PlayerEnvironment

from typing import Tuple, ClassVar, List

import numpy as np


class ConnectEnvironment(Abstract2PlayerEnvironment):

    action_space: ClassVar[Tuple[Action]] = tuple(range(7))
    directions: ClassVar[Tuple[Tuple[int, int]]] = ((1, 0), (0, 1), (1, 1), (1, -1))

    def __init__(self, player: int):
        super(ConnectEnvironment, self).__init__(player)
        self.board: State = np.zeros((7, 6, 2), dtype=np.int)
        self.reset()

    def reset(self):
        self.board[:, :, :] = 0
        self.closed = False
        self.turn = 0

    def get_state_copy(self) -> State:
        return self.board.copy()

    def get_state_with_action(self, state: State, action: Action) -> State:
        tmp, self.board = self.board, state
        self._push_action_(action)
        self.board = tmp
        return state

    def get_flipped_state_copy(self) -> State:
        return self.board.copy()[:, :, ::-1]

    def get_flipped_state_with_action(self, state: State, action: Action) -> State:
        tmp, self.board = self.board, state
        self._push_action_(action)
        self.board = tmp
        return state[:, :, ::-1]

    def get_possible_actions(self) -> List[Action]:
        return [x for x in ConnectEnvironment.action_space if self.board[x, -1, 0] == 0 and self.board[x, -1, 1] == 0]

    def is_closed(self) -> bool:
        return self.closed

    def __top__(self, x: int) -> int:
        for y in range(6):
            if self.board[x, y, 0] == 0 and self.board[x, y, 1] == 0:
                return y
        return -1

    def __count_dir__(self, x: int, y: int, player: int, vx: int, vy: int) -> int:
        count = 1
        for i in range(1, 4):
            nx, ny = x + i * vx, y + i * vy
            if nx >= 0 and nx < 7 and ny >= 0 and ny < 6 and self.board[nx, ny, player] == 1:
                count += 1
            else:
                break

        for i in range(1, 4):
            nx, ny = x - i * vx, y - i * vy
            if nx >= 0 and nx < 7 and ny >= 0 and ny < 6 and self.board[nx, ny, player] == 1:
                count += 1
            else:
                break

        return count

    def _push_action_(self, action: Action):
        if self.closed:
            raise Exception("Fatal error: game is already closed !")
        turn = self.turn
        y = self.__top__(action)
        if y == -1:
            raise Exception("Fatal error: invalid action !")
        self.board[action, y, turn] = 1

    def _check_is_closed_from_action_(self, action: Action):
        y = self.__top__(action)
        if y == -1:
            y = self.board.shape[1] - 1
        else:
            y -= 1
        for (vx, vy) in ConnectEnvironment.directions:
            if self.__count_dir__(action, y, self.turn, vx, vy) >= 4:
                self.closed = True
                self.winner = self.turn
                break

        self.closed |= np.sum(self.board) == 42


def game_to_string(state: State, player: int) -> str:
    s = "| " + " | ".join([str(x) for x in range(state.shape[0])]) + " |\n"
    s += "-" * (4 + (3 * (state.shape[0] - 1)) + state.shape[0]) + "\n"
    for y in reversed(range(state.shape[1])):
        s += "| "
        for x in range(state.shape[0]):
            if state[x, y, player] == 1:
                s += "X | "
            elif state[x, y, 1 - player] == 1:
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
