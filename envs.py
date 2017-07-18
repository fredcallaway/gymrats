from collections import namedtuple, defaultdict, deque
import numpy as np
import sys
from io import StringIO
import json
import itertools as it
import json

import gym
from gym import spaces
from gym.envs.toy_text.discrete import DiscreteEnv
from scipy.stats.distributions import bernoulli, norm

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


EMPTY = 0
WALL = 1
GOAL = 2



# One possible result of an action
# p: probability of this result
# s1: next state
# r: reward
# done: True if the episode is over else False
Result = namedtuple('Result', ['p', 's1', 'r', 'done'])

class ContextualBandit(gym.Env):
    """A one-step bandit problem with clues."""
    def __init__(self, n_feature, n_arm, payout_noise=0, weight_noise=0):
        self.n_feature = n_feature
        self.n_arm = n_arm
        self.payout_noise = norm(payout_noise)
        self.weight_noise = norm(weight_noise)
        self.w = np.random.normal(size=(self.n_feature, self.n_arm))
        self.action_space = spaces.Discrete(n_arm)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=n_feature)
        self.final_state = np.full(n_feature, np.nan)
        self._reset()

    def _reset(self):
        self._state = np.random.randn(self.n_feature)
        return self._state

    def _step(self, action):
        w = self.w + self.weight_noise.rvs((self.n_feature, self.n_arm))
        payout = self._state @ w + self.payout_noise.rvs()
        return self.final_state, payout[action], True, {}




class ForageWorld(gym.Env):
    """Gather berries."""
    metadata = {'render.modes': ['human']}
    def __init__(self, size, n_berry, time_cost=.01):
        self.size = size
        self.n_berry = n_berry
        self.time_cost = time_cost
        self.feature_space = (size, size, n_berry+1)
        self.observation_space = spaces.Discrete(np.product(self.feature_space))
        self.action_space = spaces.Discrete(4)
        
        self.berry_locs = None
        self._state = None
        self._seed()
        self._positions = {}
        idxs = iter(range(size*size))
        for r in range(size):
            for c in range(size):
                x = np.zeros(size*size)
                x[next(idxs)] = 1
                self._positions[r, c] = tuple(x)

    def _seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def _reset(self):
        initial = (self.size//2,) * 2
        locs = set()
        while len(locs) < self.n_berry:
            loc = tuple(self.np_random.randint(0, self.size, size=2).tolist())
            if loc in ((0,0), initial):
                continue
            locs.add(loc)
        self.berry_locs = {l: i for i, l in enumerate(locs)}
        self._state = (*initial, ())
        return self._observe(self._state)

    def _observe(self, state=None):
        row, col, collected = (state or self._state)
        return (*self._positions[row, col], len(collected))

    def _step(self, a):
        row, col, collected = self._state
        if (row, col) == (0, 0):  # already done
            return self._observe(self._state), 0, True, {}

        if a == UP:
            row = max(row-1, 0)
        elif a == RIGHT:
            col = min(col+1, self.size-1)
        elif a == DOWN:
            row = min(row+1, self.size-1)
        elif a == LEFT:
            col = max(col-1,0)

        berry = self.berry_locs.get((row, col))
        if berry is not None and berry not in collected:
            collected = (*collected, berry)
        self._state = (row, col, collected)

        done = True if (row, col) == (0, 0) else False
        if done:
            reward = len(collected)
        else:
            reward = - abs(self.time_cost)

        return self._observe(self._state), reward, done, {}

    def _render(self, mode='human', close=False):
        if close:
            return
        row, col, collected = self._state

        out = [[' '] * self.size for _ in range(self.size)]
        for (x,y), i in self.berry_locs.items():
            if i in collected:
                out[x][y] = 'x'
            else:
                out[x][y] = 'O'
        out[row][col] = '*'
        out[0][0] = '#'
        print('\n'.join(''.join(line) for line in out))


class GridWorld(gym.Env):
    """Move around in a grid."""
    metadata = {'render.modes': ['human', 'array']}
    WALL = 1
    GOAL = 2
    AGENT = 3

    def __init__(self, kind='maze', size=None, spec=None, rewards=None, 
                 move_cost=0.00, crash_cost=0, goal_reward=1, goal=None):
        # self.size = size
        self.kind = kind
        self.spec = spec
        self.random_spec = spec is None
        self.size = size or spec.shape[0]
        self.shape = (size, size)
        self.move_cost = - abs(move_cost)
        self.crash_cost = - abs(crash_cost)
        self.goal_reward = goal_reward

        if rewards is None:
            self.make_rewards = lambda: -np.random.random(self.shape)
        elif callable(rewards):
            self.make_rewards = rewards
        else:
            self.make_rewards = lambda: rewards
        
        self.goal = goal
        self.one_hot = False

        self._state = None
        self._seed()
        self._positions = {}
        idxs = iter(range(size*size))
        for r in range(size):
            for c in range(size):
                x = np.zeros(size*size)
                x[next(idxs)] = 1
                self._positions[r, c] = tuple(x)


        self.observation_space = spaces.Discrete(len(self._positions))
        self.action_space = spaces.Discrete(4)
        self.reset()

    def _seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def _reset(self):
        init_func = getattr(self, '_init_{}'.format(self.kind))
        init_func()
        return self._state
    
    def _init_maze(self):
        self._state = (1, 1)
        if self.random_spec:
            m = Maze(self.size, self.size)
            spec = np.stack(m.array)
            goal = m.farthest_point
            spec[goal] = self.GOAL
            self.spec = spec
            self.goal = goal
        # return self._observe(self._state)

    def _init_pain(self):
        nr, nc = self.shape
        self.goal = (nr-3, nc-3)
        self._state = (2, 2)
        if self.random_spec:
            self.spec = np.zeros(self.shape)
            self.spec[self.goal] = self.GOAL

        self.rewards = self.make_rewards()

    def _observe(self, state=None):
        row, col = self._state if state is None else state
        if self.one_hot:
            return self._positions[row, col]
        else:
            return row, col

    def location(self, state=None):
        state = self._state if state is None else state
        return state

    # def _done(self, state=None):
    #     state = self._state if state is None else state
    #     row, col = self.location(state)
    #     return (self.spec[row, col] == self.GOAL)

    def _move(self, loc, a):
        row, col = row1, col1 = loc
        if a == UP:
            row1 = max(row-1, 0)
        elif a == RIGHT:
            col1 = min(col+1, self.size-1)
        elif a == DOWN:
            row1 = min(row+1, self.size-1)
        elif a == LEFT:
            col1 = max(col-1,0)
        return row1, col1

    def _step(self, a):
        reward = 0
        done = False
        row, col = self._move(self.location(), a)
        if self.spec[row, col] == self.WALL:
            reward = self.crash_cost
        elif self.spec[row, col] == self.GOAL:
            self._state = (row, col)
            reward = self.goal_reward
            done = True
        else:
            self._state = (row, col)
            if self.rewards is not None:
                reward += self.rewards[row, col]
            reward += self.move_cost
        return self._observe(self._state), reward, done, {}

    def _render(self, mode='human', close=False):
        if close:
            return
        from matplotlib import colors
        from matplotlib import pyplot as plt
        row, col = self.location()

        def text(row, col, txt, color='w', size=22):
            plt.text(row, col, txt, color=color, size=size, 
                     verticalalignment='center', horizontalalignment='center')

        
        if self.kind == 'pain':
            plt.imshow(self.rewards, aspect='equal', cmap='Reds_r')
            text(row, col, 'A', color='b')
            text(*self.goal, 'G', color='b')
            return

        spec = self.spec.copy()
        spec[row, col] = self.AGENT
        if mode == 'array':
            return spec

        # sns.set_palette('muted', color_codes=True)
        cdef = ['#ffffff', '#111111', '#FC4754', '#4A74FF']
        cmap = colors.ListedColormap(cdef)
        bounds = np.arange(cmap.N + 1)
        cnorm = colors.BoundaryNorm(bounds, cmap.N)

        # plt.text(2, 3, 'A', size=22, verticalalignment='center', horizontalalignment='center', color='w')
        plt.imshow(spec, aspect='equal', cmap=cmap, norm=cnorm)
      

    @classmethod
    def random_maze(cls, size):
        m = Maze(size, size)
        spec = np.stack(m.array)
        goal = m.farthest_point
        spec[goal] = cls.GOAL
        return cls(spec, init_state=(1,1), goal=goal)


from maze import Maze
        

class LinearEnv(DiscreteEnv):
    """Environment in Sutton and Barto Example 6.2"""
    def __init__(self, n_states=7, reward=1, penalty=0, slip=0):
        self.n_states = n_states
        self.n_actions = n_actions = 2
        self.reward = reward
        self.penalty = penalty

        def results(s0, a):
            if s0 in (0, self.n_states-1):
                return [Result(1, 0, 0, True)]
            s1 = s0 + [-1, 1][a]
            if s1 == 0:
                yield Result(1-slip, s1, penalty, True)
            elif s1 == n_states - 1:
                yield Result(1-slip, s1, reward, True)
            elif (0 < s1 < self.n_states):
                yield Result(1-slip, s1, 0, False)

            if slip:
                s1 = s0 + [1, -1][a]
                if s1 == 0:
                    yield Result(slip, s1, penalty, True)
                elif s1 == n_states - 1:
                    yield Result(slip, s1, reward, True)
                else:
                    yield Result(slip, s1, 0, False)

        initial_state = np.zeros(n_states)
        initial_state[n_states // 2] = 1.0
        transition = {s0: {a: list(results(s0, a)) for a in range(n_actions)}
                      for s0 in range(n_states)}

        super().__init__(self.n_states, self.n_actions, transition, initial_state)


class GridEnv(DiscreteEnv):
    """A rectangluar grid with random negative rewards and one goal state."""
    def __init__(self, n_col=10, n_row=10, start=(0, 0), goal=None):
        self.grid = self.np_random.randint(-9, 0, size=n_row * n_col).reshape(n_row, n_col)
        self.n_col = n_col
        self.n_row = n_row

        goal = goal if goal else (n_row - 1, n_col - 1)
        self.grid[goal] = 0

        def results(s0, a):
            if s0 == goal:
                return [Result(1.0, s0, 0, True)]
            row, col = self.decode(s0)

            if a == UP:
                row = max(row-1, 0)
            elif a == RIGHT:
                col = min(col+1, n_col-1)
            elif a == DOWN:
                row = min(row+1, n_row-1)
            elif a == LEFT:
                col = max(col-1,0)
            
            r = self.grid[row, col]
            done = True if r >= 0 else False
            s1 = self.encode(row, col)
            return [Result(1.0, s1, r, done)]


        n_actions = 4
        n_states = n_row * n_col
        initial_state = np.zeros_like(self.grid)
        initial_state[start] = 1.0

        transition = {s0: {a: results(s0, a) for a in range(n_actions)}
                      for s0 in range(n_states)}

        super().__init__(n_states, n_actions, transition, initial_state)

    def encode(self, row, col):
        return self.n_col * row + col

    def decode(self, s):
        return (s // self.n_col, s % self.n_col)



class DecisionTreeEnv(DiscreteEnv):
    """

    """
    metadata = {'render.modes': ['human', 'ansi']}
    default_spec = (
        (2, (
            (1, (
                (1, ()),
                (1, ()),
            )),
            (1, (
                (1, ()),
                (1, ()),
            )),
        )),
        (1, (
            (3, (
                (1, ()),
                (1, ()),
            )),
            (1, (
                (1, ()),
                (4, ()),
            )),
        )),
    )

    def __init__(self, spec=None, branch=2):
        if spec is None:
            spec = self.default_spec
        P = defaultdict(dict)
        self.states = ['S']  # start state
        self.spec = spec

        q = deque()  # queue of states to expand
        def expand(spec, s0, path):
            for choice, _ in enumerate(spec):
                reward, result = _
                s1 = len(self.states)
                new_path = path + str(choice)
                self.states.append(new_path)
                done = not bool(result)
                P[s0][choice] = [(1, s1, reward, done)]
                if not done:
                    q.append((result, s1, new_path))

        q.append((spec, 0, ''))
        while q:
            expand(*q.popleft())
        isd = np.zeros(len(self.states))
        isd[0] = 1
        super().__init__(len(self.states), branch, dict(P), isd)

    @classmethod
    def random(cls, depth, branch=2):
        
        def reward(depth):
            return int(self.np_random.rand() * depth ** 2)

        def expand(depth):
            if depth:
                return [(reward(depth), expand(depth-1)) for _ in range(branch)]
            else:
                return ()

        spec = expand(depth)
        return cls(spec, branch)

    def state_name(self, state):
        return self.states[state]

    def level(self, state):
        name = self.state_name(state)
        if name == 'S':
            return 0
        else:
            return len(name)



    def _render(self, mode='human', close=False):

        if close:
            return

        outfile = StringIO() if mode == 'ansi' else sys.stdout
        def write(*args, sep=' '):
            outfile.write(sep.join(map(str, args)) + '\n')

        #desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)

        state_name = self.state_name(self.s)
        write(self.lastaction, '->', state_name)

        return outfile


class DeterministicGraphEnv(DiscreteEnv):
    """An environment specified by a graph."""
    default_spec = {
        # 'initial': {'A': 1, 'B': 1, 'C': 1},
        'initial': 'A',
        'final': lambda s: 0.1,
        'graph': (
            ('A', [('B', 5), ('C', 0)]),
            ('B', [('C', 5), ('A', 0)]),
            ('C', [('A', 5), ('B', 0)]),
        )
    }
    def __init__(self, spec=None):
        if spec is None:
            spec = self.default_spec
        elif isinstance(spec, str):
            spec = self.specs[spec]
        self.spec = spec

        initial = spec['initial']
        final = spec['final']
        graph = spec['graph']

        if isinstance(final, dict):
            final_dict = final
            final = lambda s: final_dict[s]
        elif hasattr(final, '__contains__'):
            final_set = final
            final = lambda s: int(s in final_set)
        else:
            assert callable(final)



        states, all_actions = zip(*graph)
        self.state_idx = {s: i for i, s in enumerate(states)}
        n_states = len(states)
        n_actions = max(len(actions) for actions in all_actions)
        
        initial_state = np.zeros(n_states)
        initial_state[self.state_idx[initial]] = 1

        # P[s][a] == [(probability, nextstate, reward, done), ...]
        # B default, an action has no effect.
        P = {s: {a: [(1, s, 0, False)] for a in range(n_actions)}
             for s in range(n_states)}

        for s, actions in graph:
            s = self.state_idx[s]
            for a, (s1, r) in enumerate(actions):
                s1 = self.state_idx[s1]
                p_term = final(s1)
                P[s][a] = []
                if p_term < 1:
                    P[s][a].append((1 - p_term, s1, r, False))
                if p_term > 0:
                    P[s][a].append((p_term, s1, r, True))

        super().__init__(n_states, n_actions, P, initial_state)

    @property
    def n_states(self):
        return len(self.state_idx)
        
    def to_json(self, file):
        with open(file, 'w+') as f:
            json.dump(self.P, f)


    @classmethod
    def random_tree(cls, height, branch=2, reward=None):
        if reward is None:
            reward = lambda depth: self.np_random.uniform(-10, 10)
        
        q = deque()  # queue of states to expand
        graph = []   # list of (s0, [(s1, reward)])
        ids = it.count()
        final = set()
        def expand(s0, depth):
            if depth == height:
                final.add(s0)
                graph.append((s0, []))
                return
            options = []
            for s1 in range(s0 + 1, s0 + 1 + branch):
                s1 = next(ids)
                options.append((s1, reward(depth)))
                q.append((s1, depth + 1))
            graph.append((s0, options))

        q.append((next(ids), 0))
        while q:
            expand(*q.popleft())

        spec = {
            'initial': 0,
            'final': lambda s: s in final,
            'graph': graph
        }
        return cls(spec)
        # return spec


class MazeEnv(DiscreteEnv):
    """Just a maze."""
    action_names = 'up right down left'.split()
    _specs = {

    }
    metadata = {'render.modes': ['human', 'ansi', 'step']}


    def __init__(self, spec='maze_19_1.json', is_slippery=True):
        if spec.endswith('.json'):
            with open(spec) as f:
                spec = json.load(f)
        elif spec in self._specs:
            spec = self._specs[spec]
        else:
            raise ValueError('Bad spec: {}'.format(spec))

        self.goal = tuple(spec['goal'])
        self.start = tuple(spec.get('start', (1, 1)))
        M = self.map = np.array(spec['map']).astype(int)
        M[self.goal] = GOAL
        
        self.n_row, self.n_col = n_row, n_col = self.map.shape
        n_actions = 4
        n_states = n_row * n_col

        initial_state = np.zeros_like(M)
        initial_state[self.start] = 1.0

        transition = {s : {a : [] for a in range(n_actions)} for s in range(n_states)}

        def to_s(row, col):
            return row*n_col + col
            
        def inc(row, col, a):
            if a == UP:
                row = max(row-1,0)
            elif a == RIGHT:
                col = min(col+1,n_col-1)
            elif a == DOWN:
                row = min(row+1,n_row-1)
            elif a == LEFT:
                col = max(col-1,0)
            return (row, col)

        for row in range(n_row):
            for col in range(n_col):
                s = to_s(row, col)
                for a in range(n_actions):
                    if M[row, col] in (WALL, GOAL):
                        # this state should never be left.
                        # (probability, nextstate, reward, done)
                        transition[s][a].append(Result(1.0, s, 0, True))
                    else:
                        new_row, new_col = inc(row, col, a)
                        if M[new_row, new_col] == WALL:
                            transition[s][a].append(Result(1.0, s, -1, False))
                        else:
                            new_state = to_s(new_row, new_col)
                            done = M[new_row, new_col] == GOAL
                            rew = 100 if done else -1
                            transition[s][a].append(Result(1.0, new_state, rew, done))

        super().__init__(n_states, n_actions, transition, initial_state)

    def _render(self, mode='human', close=False):
        if close:
            return
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        row, col = self.s // self.n_col, self.s % self.n_col
        
        M = self.map.tolist()
        M[row][col] = 3
        
        def block(color):
            return gym.utils.colorize('  ', color, highlight=True)

        def row_to_string(row):
            colors = {0: '  ', 1: block('gray'), 2: block('red'), 3: block('blue')}
            return ''.join(map(colors.get, row))

        outfile.write('\n'.join(row_to_string(row) for row in M) + '\n')

        if self.lastaction is not None:
            outfile.write("{}\n".format(self.action_names[self.lastaction]))
        else:
            outfile.write("\n")

        return outfile





def main():
    # env = DecisionTreeEnv.random(2)
    env = DeterministicGraphEnv.random_tree(2)
    print(env['graph'])
    # env.to_json('plane.json')


if __name__ == '__main__':
    main()














