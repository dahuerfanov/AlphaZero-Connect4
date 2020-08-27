import numpy as np
import torch

from constants import ROWS, COLS, deltas, ZERO


def gameReward(s):
    done = True
    for i in range(ROWS):
        for j in range(COLS):
            if not torch.eq(s[0][i][j], ZERO):
                for k in range(len(deltas)):
                    inARow = True
                    for p in range(3):
                        if i + deltas[k][0][p] < 0 or i + deltas[k][0][p] >= ROWS:
                            inARow = False
                            break
                        if j + deltas[k][1][p] < 0 or j + deltas[k][1][p] >= COLS:
                            inARow = False
                            break
                        if not torch.eq(s[0][i][j], s[0][i + deltas[k][0][p]][j + deltas[k][1][p]]):
                            inARow = False
                            break
                    if inARow:
                        return torch.tensor(1.), True
            else:
                done = False

    return torch.tensor(0.), done


def step(s, a):
    ones = np.count_nonzero(s.numpy() == 1.)
    twos = np.count_nonzero(s.numpy() == 2.)
    for row in range(ROWS):
        if torch.eq(s[0][row][a], ZERO):
            if ones > twos:
                s[0][row][a] = 2
            else:
                s[0][row][a] = 1
            return row

    return -1


def reflect(s):
    sr = torch.zeros((1, ROWS, COLS), dtype=torch.float32)
    for row in range(ROWS):
        for col in range(COLS):
            sr[0][row][col] = s[0][row][COLS - col - 1]

    return sr


def stateToString(s):
    sStr = ""
    for i in range(ROWS):
        for j in range(COLS):
            sStr += str(int(s[0][i][j]))

    return sStr
