import torch

from constants import ROWS, COLS, deltas

def gameReward(s, ch=0):
    for i in range(ROWS):
        for j in range(COLS):
            if s[ch][i][j] != 0:
                for k in range(len(deltas)):
                    inARow = True
                    for p in range(3):
                        if i + deltas[k][0][p] < 0 or i + deltas[k][0][p] >= ROWS:
                            inARow = False
                            break
                        if j + deltas[k][1][p] < 0 or j + deltas[k][1][p] >= COLS:
                            inARow = False
                            break
                        if s[ch][i][j] != s[ch][i + deltas[k][0][p]][j + deltas[k][1][p]]:
                            inARow = False
                            break
                    if inARow:
                        return 1., True

    return 0., torch.sum(s) == ROWS * COLS


def step(s, a, ch=0):
    row = ROWS - 1
    while row >= 0:
        if s[0][row][a] + s[1][row][a] == 0:
            s[ch][row][a] = 1
            return row
        row -= 1
    return row


def reflect(s):
    return torch.flip(s.clone(), [2])

def stateToInt(s):
    a = 0
    b = 0
    for i in range(ROWS):
        for j in range(COLS):
            if s[0][i][j].item() != 0:
                a += (1 << (j + i * COLS))
            if s[1][i][j].item() != 0:
                b += (1 << (j + i * COLS))

    return (a << (ROWS * COLS)) | b