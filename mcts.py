import math

import numpy as np
import torch
from scipy.stats import dirichlet

from constants import COLS, ALPHA_N, EPS_N, C_PUCT
from game import gameReward, reflect, step, stateToInt


class MCTS:

    def __init__(self):
        self.clear()

    def clear(self):
        self.Q = dict()
        self.P = dict()
        self.N = dict()

    def search(self, s, nnet, c_puct=C_PUCT):
        """
         Returns:
            v: the negative of the value of the current state s
        """

        v, done = gameReward(s, 1)
        if done: return v

        v, P = nnet.predict(s)
        s0 = stateToInt(s)

        if not s0 in self.N:
            self.Q[s0] = [0] * COLS
            self.N[s0] = np.array([0] * COLS)
            self.P[s0] = P

            s1 = stateToInt(reflect(s))
            self.Q[s1] = self.Q[s0][::-1]
            self.N[s1] = self.N[s0][::-1]
            self.P[s1] = self.P[s0][::-1]

            return -v

        noise = dirichlet.rvs(np.array([ALPHA_N] * COLS), size=1)

        max_u, best_a, total_sqr = -1e10, -1, math.sqrt(sum(self.N[s0]))
        for a in range(COLS):
            p_exploit = (1 - EPS_N) * self.P[s0][a] + EPS_N * noise[0][a]

            if s[0][0][a] + s[1][0][a] == 0:
                u = self.Q[s0][a] + c_puct * p_exploit * total_sqr / (1 + self.N[s0][a])
                if u > max_u:
                    max_u = u
                    best_a = a
            else:
                self.Q[s0][a] = -1

        a = best_a
        row = step(s, a)
        assert (row >= 0)
        v = self.search(torch.flip(s, [0]), nnet)
        s[0][row][a] = 0  # step back

        self.Q[s0][a] = (self.N[s0][a] * self.Q[s0][a] + v) / (self.N[s0][a] + 1)
        self.N[s0][a] += 1

        return -v

    def pi(self, s, tau=1, requires_grad=False):

        s0 = stateToInt(s)
        p = self.N[s0]
        bestAs = np.array(np.argwhere(p == np.max(p))).flatten()
        bestA = np.random.choice(bestAs)
        if tau == 0:
            p = [0] * COLS
            p[bestA] = 1
            return torch.tensor(data=p, requires_grad=requires_grad, device=torch.device("cpu"))
        else:
            counts = [x ** (1. / tau) for x in p]
            counts_sum = sum(counts)
            return torch.tensor(data=[x / counts_sum for x in counts], requires_grad=requires_grad,
                                device=torch.device("cpu"))
