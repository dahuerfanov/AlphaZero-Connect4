import torch

ROWS = 6
COLS = 7
C_PUCT = 0.45
EPOCHS = 30
MOMENTUM_SGD = 0.9
LR_SGD = 2*1e-3
NUM_ITERS = 1000
NUM_EPS = 16
NUM_EPS_PIT = 21
BATCH = 32
NUM_MCTS_SIMS = 42
THRESHOLD = 0.54
ALPHA_N = 0.03
EPS_N = 0.25
MAX_GAMES_MEM = 300
SAMPLE_SIZE = 640
EPS = 1e-6
ZERO = torch.tensor(0.)
deltas = [ [ [0,0,0],[1,2,3] ], [ [1,2,3],[0,0,0] ], [ [1,2,3],[1,2,3] ], [ [-1,-2,-3],[1,2,3] ] ]