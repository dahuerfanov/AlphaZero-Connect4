import random

import torch

from NNet import NNet
from constants import ROWS, COLS, NUM_MCTS_SIMS, NUM_EPS, NUM_ITERS, MAX_GAMES_MEM, SAMPLE_SIZE, THRESHOLD, NUM_EPS_PIT, \
    ZERO
from game import gameReward, step, reflect
from mcts import MCTS


def simulateGame(nnet1, nnet2):
    s = torch.zeros((1, ROWS, COLS), dtype=torch.float32)

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
            if torch.eq(s[0][ROWS - 1][a], ZERO):
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
def get_win_percentage(nnet1, nnet2, n_rounds=NUM_EPS_PIT):
    count = 0
    for _ in range(n_rounds // 2):
        count += max(simulateGame(nnet1, nnet2), 0)

    for _ in range(n_rounds // 2, n_rounds):
        count += max(-simulateGame(nnet2, nnet1), 0)

    return count / n_rounds


def executeEpisode(nnet):
    samples_s = []
    samples_v = []

    s = torch.zeros((1, ROWS, COLS), dtype=torch.float32)
    mcts = MCTS()

    while True:
        for _ in range(NUM_MCTS_SIMS):
            mcts.search(s, nnet)
        samples_s.append(s.clone().detach())

        x = random.random()
        a = -1
        acc = 0
        while True:
            if x < acc:
                break
            a += 1
            acc += mcts.pi(s)[a]

        if not torch.eq(s[0][ROWS - 1][a], ZERO):
            a = random.choice([col for col in range(COLS) if torch.eq(s[0][ROWS - 1][col], ZERO)])
        step(s, a)
        r, done = gameReward(s)
        if done:
            for _ in range(len(samples_s)):
                samples_v.append(r.clone().detach())
                r *= -1
            samples_v = samples_v[::-1]

            last = len(samples_s)
            for i in range(last):
                samples_s.append(reflect(samples_s[i]))
            samples_dist = [mcts.pi(s_i) for s_i in samples_s]
            samples_v = samples_v * 2

            return samples_s, samples_dist, samples_v


def policyIter():
    random.seed(1)

    nnet_1 = NNet("nnet1")
    # m_state_dict = torch.load("cnn_best_model.pth")
    # nnet_1.load_state_dict(m_state_dict)
    print(nnet_1)
    nnet_2 = NNet("nnet2")
    # nnet_2.load_state_dict(m_state_dict)
    if torch.cuda.is_available():
        nnet_1 = nnet_1.cuda()
        nnet_1.criterion_p = nnet_1.criterion_p.cuda()
        nnet_1.criterion_v = nnet_1.criterion_v.cuda()

        nnet_2 = nnet_2.cuda()
        nnet_2.criterion_p = nnet_2.criterion_p.cuda()
        nnet_2.criterion_v = nnet_2.criterion_v.cuda()

    samples_s = []
    samples_dist = []
    samples_v = []
    sizes = []

    for it in range(NUM_ITERS):
        idx_new_eps = len(samples_s)
        for e in range(NUM_EPS):
            s1, s2, s3 = executeEpisode(nnet_1)
            samples_s.extend(s1)
            samples_dist.extend(s2)
            samples_v.extend(s3)
            sizes.append(len(s1))
            if len(sizes) > MAX_GAMES_MEM:
                idx_new_eps -= sizes[0]
                samples_s = samples_s[sizes[0]:]
                samples_dist = samples_dist[sizes[0]:]
                samples_v = samples_v[sizes[0]:]
                sizes = sizes[1:]

        if SAMPLE_SIZE < idx_new_eps:
            X, Y_v, Y_p = [], [], []
            for i in random.sample([i for i in range(idx_new_eps)], SAMPLE_SIZE):
                X.append(samples_s[i])
                Y_v.append(samples_v[i])
                Y_p.append(samples_dist[i])
            X = X.extend(samples_s[idx_new_eps:])
            Y_v = Y_v.extend(samples_v[idx_new_eps:])
            Y_p = Y_p.extend(samples_dist[idx_new_eps:])
        else:
            X = samples_s
            Y_v = samples_v
            Y_p = samples_dist

        print("It #: ", it)
        nnet_2.run(X, Y_v, Y_p)
        print("calculating pit rate...")
        rate = get_win_percentage(nnet_2, nnet_1)
        print("rate nnet2 vs nnet1: ", rate)
        if rate > THRESHOLD:
            nnet_1 = nnet_2
            torch.save(nnet_1, "cnn_best_model.pt")
            torch.save(nnet_1.state_dict(), "cnn_best_model.pth")
            print("saving new model!!!!!!!")
