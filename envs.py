from collections import namedtuple, defaultdict, deque
import numpy as np
import sys
from io import StringIO
import json
import itertools as it
import json

from gym import utils
from gym.envs.toy_text.discrete import DiscreteEnv
from scipy.stats.distributions import bernoulli, norm

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


EMPTY = 0
WALL = 1
GOAL = 2

def clear_screen():
    print(chr(27) + "[2J")

Result = namedtuple('Result', ['p', 's1', 'r', 'done'])

class MDP(object):
    """A markov decision process."""

    def actions(self, state):
        """Returns a list of actions available at state."""
        raise NotImplementedError()
        
    def results(self, state, action):
        """Returns a list of possible Results of taking action in state."""
        raise NotImplementedError()
    
    def T(self, s0, a, s1):
        """Probability of transitioning from state s0 to s1 by action a."""
        return sum(result.p for result in self.results(s0, a) if result.s1 == s1)

    def R(s, a):
        """Expected reward from taking action a in state s"""
        return sum(result.p * result.r for result in self.results(s, a))


class EnvMDP(MDP):
    """An MDP defined by a DiscreteEnv object."""
    def __init__(self, env):
        super().__init__()
        self.env = env

    def actions(self, state):
        return range(self.env.action_space.n)

    def results(self, state, action):
        return self.env.P[state][action]


class MazeEnv(DiscreteEnv):
    """Just a maze."""
    action_names = 'up right down left'.split()
    _specs = {

    }
    metadata = {'render.modes': ['human', 'ansi', 'step']}


    def __init__(self, spec, is_slippery=True):
        if spec.endswith('.json'):
            with open(spec) as f:
                spec = json.load(f)
        elif spec in self._specs:
            spce = self._specs[spec]
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
        # if mode == 'step':
            # input('> ')
            # clear_screen()
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        row, col = self.s // self.n_col, self.s % self.n_col
        
        M = self.map.tolist()
        M[row][col] = 3
        
        def block(color):
            return utils.colorize('  ', color, highlight=True)

        def row_to_string(row):
            colors = {0: '  ', 1: block('gray'), 2: block('red'), 3: block('blue')}
            return ''.join(map(colors.get, row))

        outfile.write('\n'.join(row_to_string(row) for row in M) + '\n')

        if self.lastaction is not None:
            outfile.write("{}\n".format(self.action_names[self.lastaction]))
        else:
            outfile.write("\n")

        return outfile


class GridMDP(MDP):
    """Two-dimensional grid."""
    def __init__(self, n_col=10, n_row=10, seed=None):
        super().__init__()
        if seed is not None:
            np.random.seed(seed)
        self.grid = np.random.randint(-9, 0, size=n_row * n_col).reshape(n_row, n_col)
        self.goal = (n_row-1, n_col-1)
        self.grid[self.goal] = 100

    def actions(self, state):
        return [UP, RIGHT, DOWN, LEFT]

    def results(self, s0, a):
        if s0 == self.goal:
            return [Result(1.0, s0, 0, True)]
        n_row, n_col = self.grid.shape
        row, col = s0

        if a == UP:
            row = max(row-1, 0)
        elif a == RIGHT:
            col = min(col+1, n_col-1)
        elif a == DOWN:
            row = min(row+1, n_row-1)
        elif a == LEFT:
            col = max(col-1,0)
        
        s1 = (row, col)
        r = self.grid[s1]
        done = True if r > 0 else False
        return [Result(1.0, s1, r, done)]


class GridEnv(DiscreteEnv):
    """docstring for GridEnv"""
    def __init__(self, n_col=10, n_row=10, start=(0, 0), goal=None, p_term=0):
        M = self.grid = np.random.randint(-9, 0, size=n_row * n_col).reshape(n_row, n_col)
        n_actions = 4
        n_states = n_row * n_col

        initial_state = np.zeros_like(M)
        initial_state[start] = 1.0

        transition = {s : {a : [] for a in range(n_actions)} for s in range(n_states)}

        def to_s(row, col):
            return row*n_col + col

        def reformat_result(result):
            return result._replace(s1=to_s(*result.s1))

        for row in range(n_row):
            for col in range(n_col):
                for a in range(n_actions):
                    results = list(map(reformat_result, mdp.results((row,col), a)))
                    transition[to_s(row, col)][a] = results


        super().__init__(n_states, n_actions, transition, initial_state)        


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
            return int(np.random.rand() * depth ** 2)

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

    specs = {
        'plane': {
            'initial': 2,
            'final': lambda s: 1 / 6,
            'graph': (
                # (0, [(5, -3), (3, -7)]),
                # (1, [(4, 3), (0, 14)]),
                # (2, [(5, 3), (1, -3)]),
                # (3, [(2, -3), (1, -7)]),
                # (4, [(3, -3), (0, 3)]),
                # (5, [(2, -7), (4, -3)]),
                (0, [ ( 5, -3 ) , (3, -7 ) ] ) ,
                (1, [ ( 0, -3  ) , (4, -7 ) ] ) ,
                (2, [ ( 1, 14  ) , (5, 3 ) ] ) ,
                (3, [ ( 2, -3 ) , (0, 3 ) ] ) ,
                (4, [ ( 3, -3 ) , (2, -7  ) ] ) ,
                (5, [ ( 4, -3 ) , (1, 3 ) ] ) ,
                # ('william', [('jone', -3), ('clark', -7)]),
                # ('smith', [('brown', 3), ('william', 14)]),
                # ('baker', [('jone', 3), ('smith', -3)]),
                # ('clark', [('baker', -3), ('smith', -7)]),
                # ('brown', [('clark', -3), ('william', 3)]),
                # ('jone', [('baker', -7), ('brown', -3)]),
            )
        },

    }
    def __init__(self, spec=None):
        if spec is None:
            spec = self.default_spec
        elif isinstance(spec, str):
            spec = self.specs[spec]
        self.spec = spec

        graph = spec['graph']
        final = spec['final']

        states, all_actions = zip(*graph)
        self.state_idx = {s: i for i, s in enumerate(states)}
        n_states = len(states)
        n_actions = max(len(actions) for actions in all_actions)
        
        initial_state = np.zeros(n_states)
        initial_state[self.state_idx[spec['initial']]] = 1

        # P[s][a] == [(probability, nextstate, reward, done), ...]
        # B default, an action has no effect.
        P = {s: {a: [(1, s, 0, False)] for a in range(n_actions)}
             for s in range(n_states)}

        for s, actions in graph:
            s = self.state_idx[s]
            for a, (s1, r) in enumerate(actions):
                s1 = self.state_idx[s1]
                p_term = final(s1)
                P[s][a] = [(1 - p_term, s1, r, False),
                           (p_term, s1, r, True)]

        super().__init__(n_states, n_actions, P, initial_state)

    def to_json(self, file):
        with open(file, 'w+') as f:
            json.dump(self.P, f)


    @classmethod
    def random_tree(cls, height, branch=2, reward=None):
        if reward is None:
            reward = lambda depth: np.random.uniform(-10, 10)
        
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



def n_plan(n, p_term=1/6):
    spec = {
        'initial': init,
        'p_termination': p_term,
        'graph': (
            (0, [(5, -3), (3, -7)]),
            (1, [(0, -3), (4, -7)]),
            (2, [(1, 14), (5, 3)]),
            (3, [(2, -3), (0, 3)]),
            (4, [(3, -3), (2, -7)]),
            (5, [(4, -3), (1, 3)]),
        )
    }


def main():
    # env = DecisionTreeEnv.random(2)
    env = DeterministicGraphEnv.random_tree(2)
    print(env['graph'])
    # env.to_json('plane.json')


if __name__ == '__main__':
    main()














