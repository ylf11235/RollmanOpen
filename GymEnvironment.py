from typing import List
import copy
import random

import torch
import gym
import numpy as np
from gym import spaces
from .board import *
from .pacman import Pacman
from .ghost import Ghost
from .gamedata import *
from .utils import *

MAX_BOARD_SIZE = 41
rng=np.random.default_rng()

class PacmanEnv(gym.Env):
    def get_network_state_dict(self):
        # 0 pac mark
        # 1/2/3 ghost 1/2/3 mark
        # 4-13  map 10
        map_feat = torch.zeros(14, 41, 41)
        x, y = self._pacman.get_coord()
        map_feat[0, x, y] = 1
        for i in range(3):
            x, y = self._ghosts[i].get_coord()
            map_feat[i+1, x, y] = 1
        for i in range(self._size):
            for j in range(self._size):
                board_mark = int(self._board[i][j])
                map_feat[board_mark+4, i, j] = 1
        
        # 0-2 , level mark
        # 3, round / 500.0
        # 4 - 8 , x/10.0, except 7 is shield. shield would be div by 4.0
        # 9 continuous / 100.0
        # 10 eaten time / 10.0
        # 11 portal avai
        scalar_feat = torch.zeros(12)
        scalar_feat[self._level] = 1
        scalar_feat[3] = float(self._round) / 500.0
        skill_status = self._pacman.get_skills_status()
        scalar_feat[4] = skill_status[0] / 10.0
        scalar_feat[5] = skill_status[1] / 10.0
        scalar_feat[6] = skill_status[2] / 10.0
        scalar_feat[7] = skill_status[3] / 5.0
        scalar_feat[8] = skill_status[4] / 10.0
        scalar_feat[9] = self._pacman_continuous_alive / 100.0
        scalar_feat[10] = self._eaten_time / 10.0
        scalar_feat[11] = self._portal_available
        return {
            'map_feat': map_feat,
            'scalar_feat': scalar_feat
        }