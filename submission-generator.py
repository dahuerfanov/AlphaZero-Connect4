from learntools.core import binder
binder.bind(globals())
from learntools.game_ai.ex4 import *

!pip install kaggle-environments==1.2.1

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline


from kaggle_environments import make, evaluate
from gym import spaces, Env

import tensorflow as tf
from tensorflow import keras


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, Flatten, Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import initializers
from tensorflow.keras.models import model_from_json

import math
from copy import copy
from scipy.stats import dirichlet



    
    
ROWS = 6
COLS = 7
C_PUCT = 0.1
EPOCHS = 100
MOMENTUM_SGD = 0.9
LR_SGD = 1e-2
NUM_ITERS = 1000
NUM_EPS = 20
BATCH = 32
NUM_MCTS_SIMS = 50
THRESHOLD = 0.5
ALPHA_N = 0.03
EPS_N = 0.25
MAX_SAMPLES_MEM = 20000
SAMPLE_SIZE = 2560
    
def getModelNnet(name):
    
    inputs = Input( shape=(ROWS, COLS), name="input" ) 
    x  = Flatten(name="flat")(inputs) 
    x  = Dense(units=128, activation="relu", name="dense1", kernel_initializer='random_uniform', bias_initializer='zeros')(x)
    x = Dropout(0.2)(x)
    x  = Dense(units=64, activation="relu", name="dense2", kernel_initializer='random_uniform', bias_initializer='zeros')(x)
    x = Dropout(0.2)(x)
    
    x1  = Dense(units=32, activation="relu", name="dense11", kernel_initializer='random_uniform', bias_initializer='zeros')(x)
    x1 = Dropout(0.1)(x1)
    x1  = Dense(units=16, activation="relu", name="dense12", kernel_initializer='random_uniform', bias_initializer='zeros')(x1)
    x1 = Dropout(0.1)(x1)
    output_v = Dense(units=1, activation="tanh", name="out_v", kernel_initializer='random_uniform', bias_initializer='zeros')(x1)
    
    x2  = Dense(units=32, activation="relu", name="dense21", kernel_initializer='random_uniform', bias_initializer='zeros')(x)
    x2 = Dropout(0.1)(x2)
    output_dist = Dense(units=COLS, activation="softmax", name="out_dist", kernel_initializer='random_uniform', bias_initializer='zeros')(x2)
    
    model =  keras.Model(inputs=inputs, outputs=[output_v, output_dist], name=name) 
    
    model.compile(optimizer=SGD(learning_rate=LR_SGD, momentum=MOMENTUM_SGD), 
                       loss={"out_v": "MSE","out_dist": "categorical_crossentropy"},
                       loss_weights={"out_v": 1.0, "out_dist": 1.0},
                       metrics=["accuracy"]
                      )
    return model



def generateSubmissionFile():
    
    my_agent = '''

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from kaggle_environments import make, evaluate

import math
from copy import copy
from scipy.stats import dirichlet
        
    '''
    
    my_agent += '''

ROWS = 6
COLS = 7
C_PUCT = 0.1
EPOCHS = 100
MOMENTUM_SGD = 0.9
LR_SGD = 1e-2
NUM_ITERS = 1000
NUM_EPS = 20
BATCH = 32
NUM_MCTS_SIMS = 50
THRESHOLD = 0.5
ALPHA_N = 0.03
EPS_N = 0.25
MAX_SAMPLES_MEM = 20000
SAMPLE_SIZE = 2560


deltas = [ [ [0,0,0],[1,2,3] ], [ [1,2,3],[0,0,0] ], [ [1,2,3],[1,2,3] ], [ [-1,-2,-3],[1,2,3] ] ]


def gameReward(s):

    done = True
    for i in range(ROWS):
        for j in range(COLS):
            if s[i][j]!=0:
                for k in range(len(deltas)):
                    inARow = True
                    for p in range(3):
                        if i + deltas[k][0][p] < 0 or i + deltas[k][0][p] >= ROWS:
                            inARow = False
                            break
                        if j + deltas[k][1][p] < 0 or j + deltas[k][1][p] >= COLS:
                            inARow = False
                            break
                        if s[i][j] != s[i + deltas[k][0][p] ][j + deltas[k][1][p] ]:
                            inARow = False
                            break
                    if inARow:
                        return 1, True
            else:
                done = False

    return 0, done


def step(s, a):
    ones = np.count_nonzero(s == 1)
    twos = np.count_nonzero(s == 2)

    for row in range(ROWS):
        if s[row][a] == 0:
            if ones>twos:
                s[row][a] = 2
            else:
                s[row][a] = 1
            return row

    return -1

def reflect(s):
    sr = s.copy()
    for row in range(ROWS):
        for col in range(COLS):
            sr[row][col] = s[row][COLS - col - 1]

    return sr

def stateToString(s):
    sStr = ""
    for i in range(ROWS):
        for j in range(COLS):
            sStr += str(s[i][j])

    return sStr

        '''
    
    fc_layers = dict()
    hidden_units = ["dense1", "dense2", "dense11", "dense12", "out_v", "dense21", "out_dist"]
    # Get all hidden layers' weights
    for i in range(len(hidden_units)):
        fc_layers[hidden_units[i]] = [
            # weights
            str(list(np.round( nnet.get_layer(hidden_units[i]).weights[0].numpy().tolist() , 10))).replace('array(', '').replace(')', '').replace(' ', '').replace('\n', ''),
            
            # bias
            str(list(np.round( nnet.get_layer(hidden_units[i]).weights[1].numpy().tolist() , 10))).replace('array(', '').replace(')', '').replace(' ', '').replace('\n', '')
        ]
    for i in range(len(hidden_units)):
        my_agent += '''
hl{}_w = np.array({}, dtype=np.float32)

'''.format(i+1, fc_layers[hidden_units[i]][0])
        my_agent += '''
hl{}_b = np.array({}, dtype=np.float32)
'''.format(i+1, fc_layers[hidden_units[i]][1])
        
    my_agent += '''
        
class MCTS:
    
    def __init__(self):
        self.clear()

    def clear(self):
        self.Q = dict()
        self.P = dict()
        self.N = dict()

    def search(self, s):

        v, done = gameReward(s)
        if done: return v

        v, P = applyLayers(s)
        s0 = stateToString(s)
        s1 = stateToString(reflect(s))


        if not s0 in self.N:
            self.Q[s0] = [0] * COLS
            self.N[s0] = np.array([0] * COLS)
            self.P[s0] = P

            self.Q[s1] = self.Q[s0][::-1]
            self.N[s1] = self.N[s0][::-1]
            self.P[s1] = self.P[s0][::-1]

            return -v


        noise = dirichlet.rvs(np.array([ALPHA_N]*COLS), size=1)

        max_u, best_a = -1e10, -1
        for a in range(COLS):
            self.P[s0][a] = (1-EPS_N)*self.P[s0][a] + EPS_N*noise[0][a]

            if s[ROWS-1][a]==0:
                u = self.Q[s0][a]  + C_PUCT * self.P[s0][a] * math.sqrt( sum(self.N[s0]) ) / (1 + self.N[s0][a])
                if u > max_u:
                    max_u = u
                    best_a = a

        a = best_a
        row = step(s, a)
        assert(row>=0)
        v = self.search(s)
        self.Q[s0][a] = (self.N[s0][a]*self.Q[s0][a] + v)/(self.N[s0][a]+1)
        self.N[s0][a] += 1

        s[row][a] = 0 #step back
        return -v


    def pi(self, s):
        s0 = stateToString(s)
        p = self.N[s0]
        return p / sum( self.N[s0] )
        
        
def applyLayers(s):
    s = np.reshape(s, (ROWS*COLS,))
    s = np.matmul(s, hl1_w) + hl1_b
    s = s * (s > 0)  # Relu
    s = np.matmul(s, hl2_w) + hl2_b
    s = s * (s > 0)
    
    s1 = np.matmul(s, hl3_w) + hl3_b
    s1 = s1 * (s1 > 0)
    s1 = np.matmul(s1, hl4_w) + hl4_b
    s1 = s1 * (s1 > 0)
    s1 = np.matmul(s1, hl5_w) + hl5_b
    s1 = np.tanh(s1)
    
    s2 = np.matmul(s, hl6_w) + hl6_b
    s2 = s2 * (s2 > 0)
    s2 = np.matmul(s2, hl7_w) + hl7_b
    # Softmax
    e_s = np.exp(s2 - np.max(s2))
    s2 = e_s/e_s.sum()
    
    return s1, s2
    
    
mcts = MCTS()

def agent(obs, config):
    grid = reflect(np.rot90(np.rot90(np.asarray(obs.board).reshape(config.rows, config.columns))))

    if np.count_nonzero(grid)<=1: mcts.clear()

    for _ in range(NUM_MCTS_SIMS):
        mcts.search(grid)

    best_a, max_pi = -1, -1e10
    pi = mcts.pi(grid)
    for a in range(COLS):
        if grid[ROWS-1][a]==0:
            if max_pi < pi[a]:
                best_a = a
                max_pi = pi[a]

    return best_a
            
    '''
    return my_agent



nnet = getModelNnet("nnet")
nnet.load_weights("../input/my-model/best_model (15).h5")
with open('submission.py', 'w') as f:
    f.write(generateSubmissionFile())

import sys
from kaggle_environments import utils
import kaggle_environments

out = sys.stdout
submission = utils.read_file("/kaggle/working/submission.py")
agent = kaggle_environments.agent.get_last_callable(submission)
sys.stdout = out

env = make("connectx", debug=True)
env.run([agent, agent])
print("Success!" if env.state[0].status == env.state[1].status == "DONE" else "Failed...")
env.render(mode="ipython")
