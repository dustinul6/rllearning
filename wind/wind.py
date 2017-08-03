import numpy as np
import pdb
import pandas as pd


class Gridworld:
    def __init__(self, shape, start, goal):
        self.shape = shape
        self.goal = goal
        self.start = start
        self.actions = ["U", "D", "R", "L"]

    def blow(self, state):
        x = state[0]
        # ipdb.set_trace()
        if x <= 2 or x == 9:
            return 0
        elif x <= 5 or x == 8:
            return 1
        elif x <= 7:
            return 2


class Nvisited:
    def __init__(self, grid):
        self.N = {}
        self.grid = grid

    def get_state_count(self, state):
        return self.N.get(tuple(state), 0)

    def incr_state(self, state):
        current_count = self.get_state_count(state)
        self.N[tuple(state)] = current_count + 1

    def to_df(self):
        X, Y = self.grid.shape
        out = np.empty((Y, X), dtype=np.int)
        for x in range(X):
            for y in range(Y):
                i = Y - y - 1
                j = x
                state = np.array([x, y])
                out[i, j] = self.get_state_count(state)

        return pd.DataFrame(out)




class Qvalue:
    def __init__(self, grid):
        self.q = {}
        self.grid = grid

    def get(self, state, action):
        assert(isfeasible(state, self.grid))
        ret = self.q.get((state[0], state[1], action), 0)
        return ret

    def set(self, state, action, value):
        assert(isfeasible(state, self.grid))
        assert(action in self.grid.actions)
        self.q[(state[0], state[1], action)] = value


def isfeasible(state, grid):
    rightbdd = state[0] <= grid.shape[0] - 1
    leftbdd = state[0] >= 0
    upperbdd = state[1] <= grid.shape[1] - 1
    lowerbdd = state[1] >= 0
    # pdb.set_trace()
    return rightbdd & leftbdd & upperbdd & lowerbdd


def move_state(state, action, grid):
    # action \in ["U", "D", "L", "R"]
    assert(isfeasible(state, grid))
    new_state = np.copy(state)
    if action == "U":
        new_state[1] += 1
    elif action == "D":
        new_state[1] -= 1
    elif action == "L":
        new_state[0] -= 1
    elif action == "R":
        new_state[0] += 1
    # pdb.set_trace()

    return new_state if isfeasible(new_state, grid) else state


def blow_up(ini_state, grid):
    blow_dist = grid.blow(ini_state)
    state = np.copy(ini_state)
    for i in range(blow_dist):
        state = move_state(state, "U", grid)

    return state


def epsilon_wieghts(epsilon, opt_id, grid):
    Nactions = len(grid.actions)
    assert(opt_id <= Nactions - 1)
    small_prob = epsilon * 1/Nactions
    large_prob = small_prob + (1 - epsilon)
    pi = np.ones(Nactions) * small_prob
    pi[opt_id] = large_prob
    return pi


def epsilon_greedy(state, Q, epsilon, grid):
    '''find the epsilon_greedy action, a_e, and its Q(state, a_e)'''
    # the output is random \in ["U", "D", "L", "R"]
    # it is distributed according to epsilon_wieghts
    opt_id, opt_q = find_optimal(state, Q, grid)
    pi = epsilon_wieghts(epsilon, opt_id, grid)
    return np.random.choice(grid.actions, p=pi)


def find_optimal(state, Q, grid):
    '''return optimal id and Q(staet, a*) '''
    def get_action_val(a): return Q.get(state, a)
    candidate_val = np.fromiter(map(get_action_val, grid.actions), np.float)
    opt_id = np.argmax(candidate_val)
    opt_q = candidate_val[opt_id]
    return opt_id, opt_q


def next_state(state, action, grid):
    '''first blow up, then move toward the direction specified by action'''
    state1 = blow_up(state, grid)
    return move_state(state1, action, grid)


def computeVPi(Q, grid):
    X, Y = grid.shape
    V = np.zeros((Y, X))  # use cartetian coordinate
    Pi = np.empty((Y, X), dtype=np.string_)
    for x in range(X):
        for y in range(Y):
            # transform to ij indexing
            i = Y - y - 1
            j = x
            state = np.array([x, y])
            opt_id, opt_q = find_optimal(state, Q, grid)
            V[i, j] = opt_q
            Pi[i, j] = grid.actions[opt_id]

    return V, Pi


def main():
    Nepisodes = 1000
    epsilon = 0.01
    learning_rate = 0.05
    step_panelty = -1
    shape = (10, 7)
    start = np.array([0, 3])
    goal = np.array([7, 3])
    grid = Gridworld(shape, start, goal)
    Q = Qvalue(grid)
    for k in range(Nepisodes):
        print(f"episode number {k + 1}")
        state = start
        epsilon = 1 / (k + 1)
        ep_action = epsilon_greedy(state, Q, epsilon, grid)
        steps = 0
        N = Nvisited(grid)
        while not np.allclose(state, goal):
            N.incr_state(state)
            steps += 1
            new_state = next_state(state, ep_action, grid)
            # notice the distiction btw opt_Q and ep_Q
            # opt_Q = Q(next_state, optimal_action)
            # ep_Q = Q(next_state, epsilon_greedy action) <= can be suboptimal
            opt_id, opt_Q = find_optimal(new_state, Q, grid)
            ep_wieghts = epsilon_wieghts(epsilon, opt_id, grid)
            new_action = np.random.choice(grid.actions, p = ep_wieghts)
            ep_Q = Q.get(new_state, new_action)
            # print(f"ep: {k}, step; {steps}, current state: {state}, ep_action:
            # {ep_action}")
            Qnext = opt_Q
            Qcurrent = Q.get(state, ep_action)
            delta = step_panelty + Qnext - Qcurrent  # Panelty applied here
            learning_rate = 1 / N.get_state_count(state)
            Qupdate = Qcurrent + learning_rate * delta
            Q.set(state, ep_action, Qupdate)
            state = new_state
            ep_action = new_action

            if steps % 10 == 0:
                print(f"Q value: {Qupdate}, delta: {delta}")

        print(f"ep: {k} took {steps} steps")
        if k % 100 == 0:
            printVPi(Q, grid)
            print(N.to_df())
            pdb.set_trace()


def printVPi(Q, grid):
    V, Pi = computeVPi(Q, grid)
    V = pd.DataFrame(V).round(3)
    Pi = pd.DataFrame(Pi)
    print("V:", V)
    print("Pi:", Pi)


if __name__ == "__main__":
    main()
