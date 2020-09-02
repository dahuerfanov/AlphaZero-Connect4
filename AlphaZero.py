import random

import torch

from NNet import NNet
from constants import ROWS, COLS, NUM_MCTS_SIMS, NUM_EPS, NUM_ITERS, MAX_GAMES_MEM, SAMPLE_SIZE, THRESHOLD, NUM_EPS_PIT
from game import gameReward, step, reflect, stateToString
from mcts import MCTS
import numpy as np


def simulateGame(nnet1, nnet2):
    s = torch.zeros((2, ROWS, COLS), dtype=torch.float32)

    mcts1 = MCTS()
    mcts2 = MCTS()

    nnet = nnet1
    mcts = mcts1

    flag = 0

    while True:
        for _ in range(NUM_MCTS_SIMS):
            mcts.search(s, nnet, C_PUCT_PLAY)

        best_a, max_pi = -1, -1e10
        pi = mcts.pi(s)
        for a in range(COLS):
            if s[0][0][a] + s[1][0][a] == 0:
                if max_pi < pi[a]:
                    best_a = a
                    max_pi = pi[a]
        a = best_a

        step(s, a)
        r, done = gameReward(s)
        if done:
            if flag:
                return float(r)
            else:
                return float(-r)

        if flag == 0:
            nnet = nnet2
            mcts = mcts2
        else:
            nnet = nnet1
            mcts = mcts1

        s = torch.flip(s, [0])
        flag = 1 - flag


# To learn more about the evaluate() function, check out the documentation here: (insert link here)
def get_win_percentage(nnet1, nnet2, n_rounds=NUM_EPS_PIT):
    nnet1.eval()
    nnet2.eval()
    count = 0
    for _ in range(n_rounds // 2):
        count += max(simulateGame(nnet1, nnet2), 0)

    for _ in range(n_rounds // 2, n_rounds):
        count += max(-simulateGame(nnet2, nnet1), 0)

    return count / n_rounds


def executeEpisode(nnet):
    nnet.eval()

    samples_s = []
    samples_v = []

    s = torch.zeros((2, ROWS, COLS), dtype=torch.float32)
    mcts = MCTS()

    while True:
        for _ in range(NUM_MCTS_SIMS):
            mcts.search(s, nnet)
        samples_s.append(s.clone())

        x = random.random()
        a = -1
        acc = 0
        while a + 1 < COLS:
            if x < acc:
                break
            a += 1
            acc += mcts.pi(s)[a]

        if s[0][0][a] + s[1][0][a] != 0:
            a = random.choice([col for col in range(COLS) if s[0][0][col] + s[1][0][col] == 0])
        step(s, a)
        r, done = gameReward(s)
        if done:
            for _ in range(len(samples_s)):
                samples_v.append(r.clone())
                r = r * (-1)
            samples_v = samples_v[::-1]

            last = len(samples_s)
            for i in range(last):
                samples_s.append(reflect(samples_s[i]))
            samples_dist = [mcts.pi(s_i) for s_i in samples_s]
            samples_v = samples_v * 2

            return samples_s, samples_dist, samples_v
        else:
            s = torch.flip(s, [0])


def policyIter(load_path=None, save_path="edge_cnn_best_model_v4.pth",
               save_path_work="edge_cnn_best_model_v4_work.pth"):
    random.seed(0)

    nnet_1 = NNet("nnet1")
    nnet_2 = NNet("nnet2")
    if load_path != None:
        m_state_dict = torch.load(load_path)
        nnet_1.load_state_dict(m_state_dict)
        nnet_2.load_state_dict(m_state_dict)
    print(nnet_1)

    samples_s = []
    samples_dist = []
    samples_v = []
    sizes = []
    stats_s_c = dict()
    stats_s_p = dict()
    stats_s_v = dict()
    stats_s_board = dict()

    for it in range(NUM_ITERS):
        idx_new_eps = len(samples_s)
        for e in range(NUM_EPS):
            s1, s2, s3 = executeEpisode(nnet_1)
            for i in range(len(s1)):
                s = stateToString(s1[i])
                if not s in stats_s_c:
                    stats_s_p[s] = s2[i]
                    stats_s_v[s] = s3[i]
                    stats_s_c[s] = 1
                    stats_s_board[s] = s1[i]
                else:
                    stats_s_p[s] = (stats_s_c[s] * stats_s_p[s] + s2[i]) / (stats_s_c[s] + 1)
                    stats_s_v[s] = (stats_s_c[s] * stats_s_v[s] + s3[i]) / (stats_s_c[s] + 1)
                    stats_s_c[s] += 1
            samples_s.extend(s1)
            samples_dist.extend(s2)
            samples_v.extend(s3)
            sizes.append(len(s1))
            if len(sizes) > MAX_GAMES_MEM:
                for i in range(sizes[0]):
                    s = stateToString(samples_s[i])
                    assert (s in stats_s_c)
                    if stats_s_c[s] == 1:
                        del stats_s_p[s]
                        del stats_s_v[s]
                        del stats_s_c[s]
                        del stats_s_board[s]
                    else:
                        stats_s_p[s] = (stats_s_c[s] * stats_s_p[s] - samples_dist[i]) / (stats_s_c[s] - 1)
                        stats_s_v[s] = (stats_s_c[s] * stats_s_v[s] - samples_v[i]) / (stats_s_c[s] - 1)
                        stats_s_c[s] -= 1
                idx_new_eps -= sizes[0]
                samples_s = samples_s[sizes[0]:]
                samples_dist = samples_dist[sizes[0]:]
                samples_v = samples_v[sizes[0]:]
                sizes = sizes[1:]

        if SAMPLE_SIZE < idx_new_eps:
            samples_unique = set()
            for i in random.sample([j for j in range(idx_new_eps)], SAMPLE_SIZE):
                samples_unique.add(stateToString(samples_s[i]))
            for i in range(idx_new_eps, len(samples_s)):
                samples_unique.add(stateToString(samples_s[i]))
        else:
            samples_unique = stats_s_c

        X, Y_v, Y_p = [], [], []
        for s in samples_unique:
            X.append(stats_s_board[s])
            Y_v.append(stats_s_v[s])
            Y_p.append(stats_s_p[s])

        print("It #: ", it)
        nnet_2.run(torch.stack(X), torch.stack(Y_v), torch.stack(Y_p))
        print("calculating pit rate...")
        rate = get_win_percentage(nnet_2, nnet_1)
        print("rate nnet2 vs nnet1: ", rate)
        if rate > THRESHOLD:
            nnet_1 = nnet_2
            torch.save(nnet_1.state_dict(), save_path)
            print("saving new model!!!!!!!")

        torch.save(nnet_2.state_dict(), save_path_work)
        nnet_2 = NNet("nnet2")
        nnet_2.load_state_dict(torch.load(save_path_work))