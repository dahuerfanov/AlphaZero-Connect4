import random

import numpy as np
import torch

from NNet import NNet
from game import gameReward, step, reflect, stateToInt
from mcts import MCTS
from utils import save_obj, load_obj


class Agent:

    def __init__(self, nn_name, device, args):
        self.nnet = NNet(nn_name, device, args)
        self.mcts = MCTS(args)


# episode simulation of agent1 vs agent2:
def simulateEpisode(agent1, agent2, args):
    s = torch.zeros(2, args.rows, args.cols, dtype=torch.float32, device=torch.device("cpu"))
    agent = agent1
    flag = 1

    while True:
        for _ in range(args.num_mcts_sims):
            agent.mcts.search(s, agent.nnet)

        best_a, max_pi = -1, -1e10
        pi = agent.mcts.pi(s, tau=0)
        for a in range(args.cols):
            if s[0][0][a] + s[1][0][a] == 0:
                if max_pi < pi[a]:
                    best_a = a
                    max_pi = pi[a]
        a = best_a

        step(s, a, 0, args)
        r, done = gameReward(s, 0, args)
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


def get_win_percentage(agent1, agent2, args):
    agent1.nnet.eval()
    agent2.nnet.eval()
    count = 0
    tot = 0
    for _ in range(args.num_eps_pit // 2):
        sg = simulateEpisode(agent1, agent2, args)
        count += max(sg, 0)
        if sg != 0:
            tot += 1

    for _ in range(args.num_eps_pit // 2, args.num_eps_pit):
        sg = simulateEpisode(agent2, agent1, args)
        count += max(-sg, 0)
        if sg != 0:
            tot += 1

    if tot == 0:
        return 0.5, 0
    else:
        return count / tot, tot


def selfPlay(agent, args):
    agent.nnet.eval()

    samples_s = []
    samples_v = []

    s = torch.zeros((2, args.rows, args.cols), dtype=torch.float32, device=torch.device("cpu"))
    step_count = 0

    while True:
        for _ in range(args.num_mcts_sims):
            agent.mcts.search(s, agent.nnet)
        samples_s.append(s.clone())

        step_count += 1
        tau = int(step_count <= args.n_threshold_exp)

        pi = agent.mcts.pi(s, tau=tau).numpy()
        a = np.random.choice(args.cols, p=pi)

        if s[0][0][a] + s[1][0][a] != 0:
            a = random.choice([col for col in range(args.cols) if s[0][0][col] + s[1][0][col] == 0])
        step(s, a, 0, args)
        r, done = gameReward(s, 0, args)
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


def policyIter(work_path, load_model_path, name_cnn_model, device, args):
    random.seed(0)
    agent1 = Agent(name_cnn_model, device, args)
    # when a model path is provided (for continuous learning), we assume stats and boards of previous simulations will
    # be provided under the same directory:
    if load_model_path != None:
        m_state_dict = torch.load(load_model_path, map_location=device)
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
        samples_s = []  # board configurations from old simulations
        samples_dist = []  # respective action distributions
        samples_v = []  # respective rewards
        sizes = []  # for each simulation episode we store the number of its board configurations
        # the keys for the following dictionaries are the int representation of the board config.:
        stats_s_c = dict()  # number of times a board confi. has appeared along the simulations in memory, for each board conf.
        stats_s_p = dict()  # aggregated simulated distribution, for each board conf.
        stats_s_v = dict()  # aggregated simulated rewards, for each board conf.
        stats_s_board = dict()  # actual representation of board configurations.
    print(agent1.nnet)

    for it in range(args.num_iters):
        idx_new_eps = len(samples_s)
        for e in range(args.num_eps):
            agent1.mcts = MCTS(args)
            s1, s2, s3 = selfPlay(agent1, args)
            for i in range(len(s1)):
                s = stateToInt(s1[i], args)
                # we don't want to add the same board configuration many times to the memory. Instead we aggregate
                # the stats for each such board configuration and for that we make use of dictionaries:
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

            # in case we exceed the maximum number of board configurations in memory, we dequeue the old ones and
            # update the stats accordingly:
            if len(sizes) > args.max_games_mem:
                for i in range(sizes[0]):
                    s = stateToInt(samples_s[i], args)
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

        # select randomly s sample of board configurations from memory:
        if args.sample_size < idx_new_eps:
            samples_unique = set()
            for i in random.sample([j for j in range(idx_new_eps)], args.sample_size):
                samples_unique.add(stateToInt(samples_s[i], args))
            for i in range(idx_new_eps, len(samples_s)):
                samples_unique.add(stateToInt(samples_s[i], args))
        else:
            samples_unique = stats_s_c

        X, Y_v, Y_p = [], [], []
        for s in samples_unique:
            X.append(stats_s_board[s])
            Y_v.append(stats_s_v[s])
            Y_p.append(stats_s_p[s])

        # save new structures and stats under the given directory:
        save_obj(samples_s, work_path, "structures/samples_s")
        save_obj(samples_dist, work_path, "structures/samples_dist")
        save_obj(samples_v, work_path, "structures/samples_v")
        save_obj(sizes, work_path, "structures/sizes")
        save_obj(stats_s_c, work_path, "structures/stats_s_c")
        save_obj(stats_s_p, work_path, "structures/stats_s_p")
        save_obj(stats_s_v, work_path, "structures/stats_s_v")
        save_obj(stats_s_board, work_path, "structures/stats_s_board")

        print("It #: ", it)
        agent2 = Agent("nnet2", device, args)
        agent2.nnet.run(X, Y_v, Y_p)  # cnn training

        agent1.mcts = MCTS(args)
        print("calculating pit rate...")
        rate, n = get_win_percentage(agent2, agent1, args)
        print("rate nnet2 vs nnet1: ", rate, n)
        if rate >= args.threshold:
            agent1.nnet = agent2.nnet
            torch.save(agent1.nnet.state_dict(), work_path + "/models/" + name_cnn_model)
            print("new model saved!!!!!!!")
