import random

import numpy as np
import torch

from NNet import NNet
from constants import ROWS, COLS, NUM_MCTS_SIMS, NUM_EPS, NUM_ITERS, MAX_GAMES_MEM, SAMPLE_SIZE, THRESHOLD, NUM_EPS_PIT, \
    N_THRESHOLD_EXP
from game import gameReward, step, reflect, stateToInt
from mcts import MCTS
from utils import save_obj, load_obj


class Agent:

    def __init__(self, nn_name, device):
        self.nnet = NNet(nn_name, device)
        self.mcts = MCTS()


def simulateGame(agent1, agent2):
    s = torch.zeros((2, ROWS, COLS), dtype=torch.float32, device=torch.device("cpu"))

    agent = agent1
    flag = 1

    while True:
        for _ in range(NUM_MCTS_SIMS):
            agent.mcts.search(s, agent.nnet)

        best_a, max_pi = -1, -1e10
        pi = agent.mcts.pi(s, tau=0)
        for a in range(COLS):
            if s[0][0][a] + s[1][0][a] == 0:
                if max_pi < pi[a]:
                    best_a = a
                    max_pi = pi[a]
        a = best_a

        step(s, a)
        r, done = gameReward(s)
        if done:
            if flag == 1:
                if r > 0:
                    return 1
                else:
                    return 0
            else:
                if r > 0:
                    return -1
                else:
                    return 0

        if flag == 1:
            agent = agent2
        else:
            agent = agent1

        s = torch.flip(s, [0])
        flag = 1 - flag


def get_win_percentage(agent1, agent2, n_rounds=NUM_EPS_PIT):
    agent1.nnet.eval()
    agent2.nnet.eval()
    count = 0
    tot = 0
    for _ in range(n_rounds // 2):
        sg = simulateGame(agent1, agent2)
        count += max(sg, 0)
        if sg != 0:
            tot += 1

    for _ in range(n_rounds // 2, n_rounds):
        sg = simulateGame(agent2, agent1)
        count += max(-sg, 0)
        if sg != 0:
            tot += 1

    if tot == 0:
        return 0.5, 0
    else:
        return count / tot, tot


def executeEpisode(agent):
    agent.nnet.eval()

    samples_s = []
    samples_v = []

    s = torch.zeros((2, ROWS, COLS), dtype=torch.float32, device=torch.device("cpu"))
    step_count = 0

    while True:
        for _ in range(NUM_MCTS_SIMS):
            agent.mcts.search(s, agent.nnet)
        samples_s.append(s.clone())

        step_count += 1
        tau = int(step_count <= N_THRESHOLD_EXP)

        pi = agent.mcts.pi(s, tau=tau).numpy()
        a = np.random.choice(COLS, p=pi)

        if s[0][0][a] + s[1][0][a] != 0:
            a = random.choice([col for col in range(COLS) if s[0][0][col] + s[1][0][col] == 0])
        step(s, a)
        r, done = gameReward(s)
        if done:
            for _ in range(len(samples_s)):
                samples_v.append(torch.tensor(r, requires_grad=False, device=torch.device("cpu")))
                r = r * (-1)
            samples_v = samples_v[::-1]

            last = len(samples_s)
            for i in range(last):
                samples_s.append(reflect(samples_s[i]))
            samples_dist = [agent.mcts.pi(s_i) for s_i in samples_s]
            samples_v = samples_v * 2

            return samples_s, samples_dist, samples_v
        else:
            s = torch.flip(s, [0])


def policyIter(work_path, load_path_model=None, name_cnn_model="nnet", device=torch.device('cpu')):
    random.seed(0)
    agent1 = Agent(name_cnn_model, device)
    if load_path_model != None:
        m_state_dict = torch.load(load_path_model, map_location=device)
        agent1.nnet.load_state_dict(m_state_dict)
        samples_s = load_obj(work_path, "structures/samples_s")
        samples_dist = load_obj(work_path, "structures/samples_dist")
        samples_v = load_obj(work_path, "structures/samples_v")
        sizes = load_obj(work_path, "structures/sizes")
        stats_s_c = load_obj(work_path, "structures/stats_s_c")
        stats_s_p = load_obj(work_path, "structures/stats_s_p")
        stats_s_v = load_obj(work_path, "structures/stats_s_v")
        stats_s_board = load_obj(work_path, "structures/stats_s_board")
    else:
        samples_s = []
        samples_dist = []
        samples_v = []
        sizes = []
        stats_s_c = dict()
        stats_s_p = dict()
        stats_s_v = dict()
        stats_s_board = dict()
    print(agent1.nnet)

    for it in range(NUM_ITERS):
        idx_new_eps = len(samples_s)
        for e in range(NUM_EPS):
            agent1.mcts = MCTS()
            s1, s2, s3 = executeEpisode(agent1)
            for i in range(len(s1)):
                s = stateToInt(s1[i])
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
                    s = stateToInt(samples_s[i])
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
                samples_unique.add(stateToInt(samples_s[i]))
            for i in range(idx_new_eps, len(samples_s)):
                samples_unique.add(stateToInt(samples_s[i]))
        else:
            samples_unique = stats_s_c

        X, Y_v, Y_p = [], [], []
        for s in samples_unique:
            X.append(stats_s_board[s])
            Y_v.append(stats_s_v[s])
            Y_p.append(stats_s_p[s])

        save_obj(samples_s, work_path, "structures/samples_s")
        save_obj(samples_dist, work_path, "structures/samples_dist")
        save_obj(samples_v, work_path, "structures/samples_v")
        save_obj(sizes, work_path, "structures/sizes")
        save_obj(stats_s_c, work_path, "structures/stats_s_c")
        save_obj(stats_s_p, work_path, "structures/stats_s_p")
        save_obj(stats_s_v, work_path, "structures/stats_s_v")
        save_obj(stats_s_board, work_path, "structures/stats_s_board")

        print("It #: ", it)
        agent2 = Agent("nnet2", device)
        agent2.nnet.run(X, Y_v, Y_p)

        agent1.mcts = MCTS()
        print("calculating pit rate...")
        rate, n = get_win_percentage(agent2, agent1)
        print("rate nnet2 vs nnet1: ", rate, n)
        if rate >= THRESHOLD:
            agent1.nnet = agent2.nnet
            torch.save(agent1.nnet.state_dict(), work_path + "/models/" + name_cnn_model)
            print("new model saved!!!!!!!")
