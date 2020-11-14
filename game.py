import torch

deltas = [[[0, 0, 0], [1, 2, 3]], [[1, 2, 3], [0, 0, 0]], [[1, 2, 3], [1, 2, 3]], [[-1, -2, -3], [1, 2, 3]]]


def gameReward(s, ch, args):
    for i in range(args.rows):
        for j in range(args.cols):
            if s[ch][i][j] != 0:
                for k in range(len(deltas)):
                    inARow = True
                    for p in range(3):
                        if i + deltas[k][0][p] < 0 or i + deltas[k][0][p] >= args.rows:
                            inARow = False
                            break
                        if j + deltas[k][1][p] < 0 or j + deltas[k][1][p] >= args.cols:
                            inARow = False
                            break
                        if s[ch][i][j] != s[ch][i + deltas[k][0][p]][j + deltas[k][1][p]]:
                            inARow = False
                            break
                    if inARow:
                        return 1., True

    return 0., torch.sum(s) == args.rows * args.cols


def step(s, a, ch, args):
    row = args.rows - 1
    while row >= 0:
        if s[0][row][a] + s[1][row][a] == 0:
            s[ch][row][a] = 1
            return row
        row -= 1
    return row


def reflect(s):
    return torch.flip(s.clone(), [2])


def stateToInt(s, args):
    a = 0
    b = 0
    for i in range(args.rows):
        for j in range(args.cols):
            if s[0][i][j].item() != 0:
                a += (1 << (j + i * args.cols))
            if s[1][i][j].item() != 0:
                b += (1 << (j + i * args.cols))

    return (a << (args.rows * args.cols)) | b
