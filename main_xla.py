# for TPU
import torch_xla.core.xla_model as xm
from dotdict import DotDict

from alphaZero import policyIter

args = DotDict({
    'rows': 6,                  # number of board rows
    'cols': 7,                  # number of board columns
    'c_puct': 1,                # Exploration coefficient in mcts
    'n_threshold_exp': 14,      # Num. of episodes at the beginning of self-play sim. for which tau = 1,afterwards tau=0
    'epochs': 50,               # Epochs for training cnn in each iteration
    'momentum_sgd': 0.95,       # momentum in sgd algorithm
    'lr_sgd': 1e-2,             # learning rate of sgd algorithm
    'wd_sgd': 4e-3,             # weight decay of sgd algorithm
    'num_iters': 1000,          # maximum number of iterations of AlphaZero algorithm
    'num_eps': 100,             # number of simulations of self-play
    'num_eps_pit': 21,          # number of simulations to pit the new and the best cnn
    'batch': 128,
    'num_mcts_sims': 24,        # number of mcts simulations
    'threshold': 0.57,          # pit rate thereshold to decide the best cnn
    'alpha_n': 1,               # alpha constant of Dirichlet Noise in mcts
    'eps_n': 0.25,              # eps constant of Dirichlet Noise in mcts
    'max_games_mem': 2000,      # max. umber of episodes in memory to train new cnn's
    'sample_size': 20000,       # board sample size to choose from old episodes in memory
    'num_channels_cnn': 128
})

device = device = xm.xla_device()
print("Using TPU!")
policyIter(work_path="data/", load_model_path=None, name_cnn_model="cnn_v128_model", device=device, args=args)
