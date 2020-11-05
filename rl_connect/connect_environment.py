from rl_connect.abstract_environment import AbstractEnvironment, Action, State

from typing import Tuple, ClassVar, List

import numpy as np


class ConnectEnvironment(AbstractEnvironment):

    action_space: ClassVar[Tuple[Action]] = tuple(range(7))
    directions: ClassVar[Tuple[Tuple[int, int]]] = ((1, 0), (0, 1), (1, 1), (1, -1))

    def __init__(self, player: int):
        self.board: State = np.zeros((7, 6, 2), dtype=np.int)
        self.player: int = player
        self.reset()

    def reset(self):
        self.board[:, :, :] = 0
        self.closed = False
        self.turn = 0

    def get_state_copy(self) -> State:
        return self.board.copy()

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

    def do_action(self, action: Action):
        if self.closed:
            raise Exception("Fatal error: game is already closed !")
        turn = self.turn
        self.turn = 1 - self.turn
        y = self.__top__(action)
        if y == -1:
            raise Exception("Fatal error: invalid action !")
        self.board[action, y, turn] = 1
        for (vx, vy) in ConnectEnvironment.directions:
            if self.__count_dir__(action, y, turn, vx, vy) >= 4:
                self.closed = True
        if self.closed:
            return 1 if self.player == turn else -1
        self.closed |= np.sum(self.board) == 42
        return 0
