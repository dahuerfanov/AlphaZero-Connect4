#!/usr/bin/env python
# coding: utf-8

# In[5]:


from learntools.core import binder

binder.bind(globals())
from learntools.game_ai.ex4 import *

get_ipython().system('pip install kaggle-environments')

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

from gym import spaces, Env

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Input, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import initializers

import math
from copy import copy
from scipy.stats import dirichlet

# In[6]:


ROWS = 6
COLS = 7
C_PUCT = 0.1
EPOCHS = 100
MOMENTUM_SGD = 0.9
LR_SGD = 1e-3
NUM_ITERS = 1000
NUM_EPS = 16
BATCH = 32
NUM_MCTS_SIMS = 100
THRESHOLD = 0.55
ALPHA_N = 0.03
EPS_N = 0.25
MAX_SAMPLES_MEM = 40000
SAMPLE_SIZE = 5120

deltas = [[[0, 0, 0], [1, 2, 3]], [[1, 2, 3], [0, 0, 0]], [[1, 2, 3], [1, 2, 3]], [[-1, -2, -3], [1, 2, 3]]]


def gameReward(s):
    done = True
    for i in range(ROWS):
        for j in range(COLS):
            if s[i][j] != 0:
                for k in range(len(deltas)):
                    inARow = True
                    for p in range(3):
                        if i + deltas[k][0][p] < 0 or i + deltas[k][0][p] >= ROWS:
                            inARow = False
                            break
                        if j + deltas[k][1][p] < 0 or j + deltas[k][1][p] >= COLS:
                            inARow = False
                            break
                        if s[i][j] != s[i + deltas[k][0][p]][j + deltas[k][1][p]]:
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
            if ones > twos:
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


class MCTS:

    def __init__(self):
        self.clear()

    def clear(self):
        self.Q = dict()
        self.P = dict()
        self.N = dict()

    def search(self, s, nnet):

        v, done = gameReward(s)
        if done: return v

        _v, _P = nnet.predict(np.expand_dims(s, axis=0))
        P = _P[0]
        v = _v[0]
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

        noise = dirichlet.rvs(np.array([ALPHA_N] * COLS), size=1)

        max_u, best_a = -1e10, -1
        for a in range(COLS):
            self.P[s0][a] = (1 - EPS_N) * self.P[s0][a] + EPS_N * noise[0][a]

            if s[ROWS - 1][a] == 0:
                u = self.Q[s0][a] + C_PUCT * self.P[s0][a] * math.sqrt(sum(self.N[s0])) / (1 + self.N[s0][a])
                if u > max_u:
                    max_u = u
                    best_a = a

        a = best_a
        row = step(s, a)
        assert (row >= 0)
        v = self.search(s, nnet)
        self.Q[s0][a] = (self.N[s0][a] * self.Q[s0][a] + v) / (self.N[s0][a] + 1)
        self.N[s0][a] += 1

        s[row][a] = 0  # step back
        return -v

    def pi(self, s):
        s0 = stateToString(s)
        p = self.N[s0]
        return p / sum(self.N[s0])


# In[7]:


def getModelNnet(name):
    inputs = Input(shape=(ROWS, COLS), name="input")
    x = Flatten(name="flat")(inputs)
    x = Dense(units=128, activation="relu", name="dense1", kernel_initializer='random_uniform',
              bias_initializer='zeros')(x)
    x = Dropout(0.2)(x)
    x = Dense(units=64, activation="relu", name="dense2", kernel_initializer='random_uniform',
              bias_initializer='zeros')(x)
    x = Dropout(0.2)(x)

    x1 = Dense(units=32, activation="relu", name="dense11", kernel_initializer='random_uniform',
               bias_initializer='zeros')(x)
    x1 = Dropout(0.2)(x1)
    x1 = Dense(units=16, activation="relu", name="dense12", kernel_initializer='random_uniform',
               bias_initializer='zeros')(x1)
    x1 = Dropout(0.2)(x1)
    output_v = Dense(units=1, activation="tanh", name="out_v", kernel_initializer='random_uniform',
                     bias_initializer='zeros')(x1)

    x2 = Dense(units=32, activation="relu", name="dense21", kernel_initializer='random_uniform',
               bias_initializer='zeros')(x)
    x2 = Dropout(0.1)(x2)
    output_dist = Dense(units=COLS, activation="softmax", name="out_dist", kernel_initializer='random_uniform',
                        bias_initializer='zeros')(x2)

    model = keras.Model(inputs=inputs, outputs=[output_v, output_dist], name=name)
    model.compile(optimizer=SGD(learning_rate=LR_SGD, momentum=MOMENTUM_SGD),
                  loss={"out_v": "MSE", "out_dist": "categorical_crossentropy"},
                  loss_weights={"out_v": 1.0, "out_dist": 1.0},
                  metrics=["accuracy"]
                  )

    return model


def simulateGame(nnet1, nnet2):
    s = np.zeros((ROWS, COLS), dtype=int)

    mcts1 = MCTS()
    mcts2 = MCTS()

    nnet = nnet1
    mcts = mcts1

    flag = 0

    while True:
        for _ in range(NUM_MCTS_SIMS):
            mcts.search(s, nnet)

        best_a, max_pi = -1, -1e10
        pi = mcts.pi(s)
        for a in range(COLS):
            if s[ROWS - 1][a] == 0:
                if max_pi < pi[a]:
                    best_a = a
                    max_pi = pi[a]
        a = best_a

        step(s, a)
        r, done = gameReward(s)
        if done:
            if flag:
                return r
            else:
                return -r

        if flag == 0:
            nnet = nnet2
            mcts = mcts2
        else:
            nnet = nnet1
            mcts = mcts1

        flag = 1 - flag


# To learn more about the evaluate() function, check out the documentation here: (insert link here)
def get_win_percentage(nnet1, nnet2, n_rounds=NUM_EPS):
    count = 0
    for _ in range(n_rounds // 2):
        count += max(simulateGame(nnet1, nnet2), 0)

    for _ in range(n_rounds // 2, n_rounds):
        count += max(-simulateGame(nnet2, nnet1), 0)

    return count / n_rounds


def executeEpisode(nnet):
    samples_s = []
    samples_v = []

    s = np.zeros((ROWS, COLS), dtype=int)
    mcts = MCTS()

    while True:
        for _ in range(NUM_MCTS_SIMS):
            mcts.search(s, nnet)
        samples_s.append(s.copy())

        x = random.random()
        a = -1
        acc = 0
        while True:
            if x < acc:
                break
            a += 1
            acc += mcts.pi(s)[a]

        if s[ROWS - 1][a] != 0:
            a = random.choice([col for col in range(COLS) if s[ROWS - 1][col] == 0])
        step(s, a)
        r, done = gameReward(s)
        if done:
            for _ in range(len(samples_s)):
                samples_v.append(r)
                r *= -1
            samples_v = samples_v[::-1]

            last = len(samples_s)
            for i in range(last):
                samples_s.append(reflect(samples_s[i]))
            samples_dist = [mcts.pi(s_i) for s_i in samples_s]
            samples_v = samples_v * 2

            return samples_s, samples_dist, samples_v


def policyIter():
    nnet_1 = getModelNnet("nnet1")
    nnet_1.load_weights("../input/my-model/best_model.h5")
    nnet_2 = getModelNnet("nnet2")
    nnet_2.load_weights("../input/my-model/best_model.h5")

    random.seed(1)

    samples_s = []
    samples_dist = []
    samples_v = []

    for it in range(NUM_ITERS):
        for e in range(NUM_EPS):
            s1, s2, s3 = executeEpisode(nnet_1)
            samples_s.extend(s1)
            samples_dist.extend(s2)
            samples_v.extend(s3)

        data_size = len(samples_s)
        samples_s = samples_s[max(data_size - MAX_SAMPLES_MEM, 0):]
        samples_dist = samples_dist[max(data_size - MAX_SAMPLES_MEM, 0):]
        samples_v = samples_v[max(data_size - MAX_SAMPLES_MEM, 0):]

        data_size = len(samples_s)
        if SAMPLE_SIZE < data_size:
            X = np.empty(SAMPLE_SIZE)
            Y_v = np.empty(SAMPLE_SIZE)
            Y_p = np.empty(SAMPLE_SIZE)
            ind_sample = random.sample([i for i in range(data_size)], SAMPLE_SIZE)
            X, Y_v, Y_p = [], [], []
            for i in ind_sample:
                X.append(samples_s[i])
                Y_v.append(samples_v[i])
                Y_p.append(samples_dist[i])
            X = np.array(X)
            Y_v = np.array(Y_v)
            Y_p = np.array(Y_p)
        else:
            X = np.array(samples_s)
            Y_v = np.array(samples_v)
            Y_p = np.array(samples_dist)

        history = nnet_2.fit(x=X, y={"out_v": Y_v, "out_dist": Y_p}, batch_size=BATCH, epochs=EPOCHS, shuffle=True,
                             verbose=0, validation_split=0.2).history

        # Plot history: loss out_v
        plt.plot(history['out_v_loss'], label='loss (training data)')
        plt.plot(history['val_out_v_loss'], label='loss (validation data)')
        plt.title('Loss functions V')
        plt.ylabel('loss value')
        plt.xlabel('No. epoch')
        plt.legend(loc="upper left")
        plt.show()
        # Plot history: loss out_dist
        plt.plot(history['out_dist_loss'], label='loss (training data)')
        plt.plot(history['val_out_dist_loss'], label='loss (validation data)')
        plt.title('Loss functions Dist')
        plt.ylabel('loss value')
        plt.xlabel('No. epoch')
        plt.legend(loc="upper left")
        plt.show()

        # Plot history: accuracy out_v
        plt.plot(history['out_v_accuracy'], label='accuracy (training data)')
        plt.plot(history['val_out_v_accuracy'], label='accuracy (validation data)')
        plt.title('Accuracy V')
        plt.ylabel('accuracy')
        plt.xlabel('No. epoch')
        plt.legend(loc="upper left")
        plt.show()
        # Plot history: accuracy out_dist
        plt.plot(history['out_dist_accuracy'], label='accuracy (training data)')
        plt.plot(history['val_out_dist_accuracy'], label='accuracy (validation data)')
        plt.title('Accuracy Dist')
        plt.ylabel('accuracy')
        plt.xlabel('No. epoch')
        plt.legend(loc="upper left")
        plt.show()

        print("It #: ", it)
        rate = get_win_percentage(nnet_2, nnet_1)
        print("rate nnet2 vs nnet1: ", rate)
        if rate > THRESHOLD:
            nnet_1 = nnet_2
            nnet_1.save_weights("best_model.h5")
            print("saving new model!!!!!!!!!!!!!!")

    return nnet_1


# In[8]:


nnet = policyIter()
# serialize model to JSON
model_json = nnet.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
nnet.save_weights("model.h5")
print("Saved model to disk")
