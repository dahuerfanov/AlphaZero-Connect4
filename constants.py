import torch

ROWS = 6
COLS = 7
C_PUCT = 0.25
EPOCHS = 50
MOMENTUM_SGD = 0.9
LR_SGD = 1e-3
NUM_ITERS = 1000
NUM_EPS = 25
NUM_EPS_PIT = 20
BATCH = 32
NUM_MCTS_SIMS = 32
THRESHOLD = 0.54
ALPHA_N = 0.03
EPS_N = 0.25
MAX_GAMES_MEM = 200000
SAMPLE_SIZE = 25600
EPS = 1e-6
ZERO = torch.tensor(0.)
deltas = [[[0, 0, 0], [1, 2, 3]], [[1, 2, 3], [0, 0, 0]], [[1, 2, 3], [1, 2, 3]], [[-1, -2, -3], [1, 2, 3]]]
