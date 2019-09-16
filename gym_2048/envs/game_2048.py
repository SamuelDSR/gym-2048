import sys
from io import StringIO

import gym
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding

from gym_2048.engine import Engine


class Game2048(gym.Env):
    metadata = {'render.modes': ['human', 'ansi']}

    def flatten(self, l):
        return [item for sublist in l for item in sublist]

    def __init__(self, seed=None):
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(16 * 4 * 4)
        self.reward_range = (0, np.inf)
        if seed is None:
            self.seed = seeding.create_seed()
        else:
            self.seed = seed
        self.env = Engine(seed=self.seed)
        self.env.reset_game()

    def step(self, action):
        assert self.action_space.contains(action)

        reward, ended = self.env.move(action)
        return self.env.get_board(), reward, ended, {
            'score': self.env.score,
            'won': self.env.won
        }

    def reset(self):
        self.env.reset_game()
        return self.env.get_board()

    def render(self, mode='human', close=False):
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        outfile.write(str(self.env))

    def moves_available(self):
        return self.env.moves_available()

    def onehot_2s(self, val):
        zeros = np.zeros(16)
        if val == 0:
            zeros[0] = 1
        else:
            zeros[int(np.log2(val))] = 1
        return zeros

    def onehot_board(self, board):
        bin_board = np.zeros(4, 4, 16)
        for r in range(0, 4):
            for c in range(0, 4):
                val = board[r, c]
                if val == 0:
                    bin_board[r, c, 0] = 1
                else:
                    bin_board[r, c, int(np.log2(val))] = 1
        return bin_board.reshape(-1)
