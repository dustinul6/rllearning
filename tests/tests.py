import unittest
import numpy as np
import wind.wind as wind
import pdb
import numpy.testing as npt


class TestEnv(unittest.TestCase):
    def setUp(self):
        shape = (10, 7)
        start = np.array([0, 3])
        goal = np.array([9, 3])
        state1 = np.array([1, 1])
        state2 = np.array([1, 2])
        state3 = np.array([2, 1])
        self.grid1 = wind.Gridworld(shape, start, goal)
        self.state1 = state1
        self.Q = wind.Qvalue(self.grid1)
        self.Q.set(state1, "U", 0.3)
        self.Q.set(state1, "D", 0.9)
        self.Q.set(state1, "L", 1.3)
        self.Q.set(state2, "R", 0.6)
        self.Q.set(state1, "D", 0.8)

    def test_feasible(self):
        s1 = np.array([7, 3])
        self.assertTrue(wind.isfeasible(s1, self.grid1))
        s2 = np.array([5, 7])
        self.assertFalse(wind.isfeasible(s2, self.grid1))

    def test_move_state(self):
        state = np.array([0, 0])
        new_state = wind.move_state(state, "U", self.grid1)
        npt.assert_allclose(new_state, np.array([0, 1]))
        new_state = wind.move_state(state, "D", self.grid1)
        npt.assert_allclose(new_state, state)
        state = np.array([5, 6])
        new_state = wind.move_state(state, "U", self.grid1)
        npt.assert_allclose(new_state, state)
        state = np.array([9, 2])
        new_state = wind.move_state(state, "R", self.grid1)
        npt.assert_allclose(new_state, state)
        new_state2 = wind.move_state(state, "U", self.grid1)
        npt.assert_allclose(new_state2, np.array([9, 3]))

    # def test_blow_dist(self):
        # state = np.array([1,1])
        # self.grid1.blow(state)

    def test_blow(self):
        new_state = wind.blow_up(self.state1, self.grid1)
        npt.assert_allclose(new_state, np.array([1, 1]))
        s1 = np.array([5, 4])
        s2 = wind.blow_up(s1, self.grid1)
        npt.assert_allclose(s2, np.array([5, 5]))
        s1 = np.array([6, 4])
        s2 = wind.blow_up(s1, self.grid1)
        npt.assert_allclose(s2, np.array([6, 6]))

    #def test_epsilon_prob2(self):
        #e = 0.001
        #opt_id = 2
        #pi_1 = wind.epsilon_wieghts(e, opt_id, self.grid1)
        #pi_target1 = np.zeros(4, np.float)
        #pi_target1[opt_id] = 1 - e + e * 1/4
        #sum_pi1 = pi_1.sum()
        #sum_target = pi_target1.sum()
        #npt.assert_allclose(pi_target1, pi_1)
        #npt.assert_allclose(sum_pi1, 1.0)
        #npt.assert_allclose(sum_target, 1.0)

    def test_epsilon_prob1(self):
        e = 1.0
        opt_id = 2
        pi_1 = wind.epsilon_wieghts(e, opt_id, self.grid1)
        pi_target1 = np.ones(4) * 0.25
        sum_pi1 = pi_1.sum()
        sum_target = pi_target1.sum()
        npt.assert_allclose(pi_target1, pi_1)
        npt.assert_allclose(sum_pi1, 1.0)
        npt.assert_allclose(sum_target, 1.0)

    def test_next_state(self):
        s1 = np.array([6, 4])
        s2 = wind.next_state(s1, "U", self.grid1)
        target = np.array([6, 6])
        npt.assert_allclose(s2, target)
        s1 = np.array([5, 2])
        s2 = wind.next_state(s1, "R", self.grid1)
        target = np.array([6, 3])
        npt.assert_allclose(s2, target)
        s1 = np.array([7, 2])
        s2 = wind.next_state(s1, "R", self.grid1)
        target = np.array([8, 4])
        npt.assert_allclose(s2, target)
        s1 = np.array([8, 6])
        s2 = wind.next_state(s1, "L", self.grid1)
        target = np.array([7, 6])
        npt.assert_allclose(s2, target)

   # def test_opt_policy(sefl):

    def test_epsilon_prob(self):
        e = 0.05
        opt_id = 2
        pi_1 = wind.epsilon_wieghts(e, opt_id, self.grid1)
        pi_target1 = np.array([0.0125, 0.0125, 0.95 + 0.0125, 0.0125])
        sum_pi1 = pi_1.sum()
        sum_target = pi_target1.sum()
        npt.assert_allclose(pi_target1, pi_1)
        npt.assert_allclose(sum_pi1, 1.0)
        npt.assert_allclose(sum_target, 1.0)

    def test_epsilon_greedy(self):
        e = 0.01
        N = 1000
        dist = np.zeros(4, int)
        state = self.state1
        # U:0, D:1, L:2, R:3
        for i in range(N):
            a = wind.epsilon_greedy(state, self.Q, e, self.grid1)
            if a == "U":
                dist[0] += 1
            elif a == "D":
                dist[1] += 1
            elif a == "L":
                dist[2] += 1
            elif a == "R":
                dist[3] += 1
            else:
                raise Exception('unknown action')
        print(f"action dist: {dist}")
        pdb.set_trace()
        self.assertTrue(True)


class TestQ(unittest.TestCase):
    def setUp(self):
        shape = (7, 10)
        start = np.array([0, 3])
        goal = np.array([7, 3])
        state1 = np.array([1, 1])
        state2 = np.array([1, 2])
        state3 = np.array([2, 2])
        grid = wind.Gridworld(shape, start, goal)
        self.grid = grid
        self.Q = wind.Qvalue(grid)
        self.states = [state1, state2, state3]

    def test_get(self):
        # default values
        v0 = self.Q.get(self.states[0], "U")
        v1 = self.Q.get(self.states[1], "R")
        npt.assert_allclose(v0, 0.0)
        npt.assert_allclose(v1, 0.0)

    def test_set(self):
        s1 = self.states[0]
        self.Q.set(s1, "U", 0.5)
        v1 = self.Q.get(s1, "U")
        npt.assert_allclose(v1, 0.5)


class TestPath(unittest.TestCase):
    def setUp(self):
        shape = (10, 7)
        start = np.array([0, 3])
        goal = np.array([7, 3])
        state1 = np.array([1, 1])
        state2 = np.array([1, 2])
        state3 = np.array([2, 1])
        self.grid1 = wind.Gridworld(shape, start, goal)
        self.state1 = state1
        self.Q = wind.Qvalue(self.grid1)

    def test_path(self):
        policy = "R" * 9  + "D" * 4 + "L" * 2
        policy = list(policy)
        state = self.grid1.start
        for action in policy:
            next_state = wind.next_state(state, action, self.grid1)
            # pdb.set_trace()
            state = next_state

        npt.assert_allclose(state, self.grid1.goal)



if __name__ == '__main__':
    unittest.main()
