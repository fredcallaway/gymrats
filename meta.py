from collections import namedtuple, defaultdict, deque, Counter
import numpy as np
from utils import PriorityQueue
from agents import Agent, Model
import gym
from gym import spaces
from policies import Policy
from toolz import memoize, curry
import itertools as it

from utils import log_return
from distributions import *
# from envs import 

class MetaBestFirstSearchEnv(gym.Env):
    """A meta-MDP for best first search with a deterministic transition model."""
    Node = namedtuple('Node', ('state', 'path', 'reward', 'done'))
    State = namedtuple('State', ('frontier', 'reward_to_state', 'best_done'))
    TERM = 'TERM'

    def __init__(self, env, eval_node, expansion_cost=0.01):
        super().__init__()
        self.env = env
        self.expansion_cost = - abs(expansion_cost)

        # This guy interacts with the external environment, what a chump!
        self.surface_agent = Agent()
        self.surface_agent.register(self.env)
        self.eval_node = eval_node

    def _reset(self):
        self.env.reset()
        self.model = Model(self.env)  # warning: this breaks if env resets again
        start = self.Node(self.env._state, [], 0, False)
        frontier = PriorityQueue(key=self.eval_node(noisy=True))  # this is really part of the Meta Policy
        frontier.push(start)
        reward_to_state = defaultdict(lambda: -np.inf)
        best_done = None
        # Warning: state is mutable (and we mutate it!)
        self._state = self.State(frontier, reward_to_state, best_done)
        return self._state

    def _step(self, action):
        """Expand a node in the frontier."""
        if action is self.TERM:
            # The return of one episode in the external env is
            # one reward in the MetaSearchEnv.
            trace = self._execute_plan()
            external_reward = trace['return']
            return None, external_reward, True, {'trace': trace}
        else:
            return self._expand_node(action), self.expansion_cost, False, {}

    def _execute_plan(self):
        frontier, reward_to_state, best_done = self._state

        if not best_done:
            raise RuntimeError('Cannot make plan.')

        policy = FixedPlanPolicy(best_done.path)
        self.surface_agent.register(policy)
        trace = self.surface_agent.run_episode(reset=False)
        return trace

        # elif frontier:
        #     plan = min(best_done, frontier.pop(), key=eval_node)
        #     plan = frontier.pop()

    def _expand_node(self, node):
        frontier, reward_to_state, best_done = self._state
        s0, p0, r0, _ = node

        for a, s1, r, done in self.model.options(s0):
            node1 = self.Node(s1, p0 + [a], r0 + r, done)
            if node1.reward <= reward_to_state[s1] -0.002:
                continue  # cannot be better than an existing node
            reward_to_state[s1] = node1.reward
            if done:
                best_done = max((best_done, node1), key=self.eval_node(noisy=False))
            else:
                frontier.push(node1)

        self._state = self.State(frontier, reward_to_state, best_done)
        return self._state


class FixedPlanPolicy(Policy):
    """A policy that blindly executes a fixed sequence of actions."""
    Node = namedtuple('Node', ('state', 'path', 'reward', 'done'))
    def __init__(self, plan, **kwargs):
        super().__init__(**kwargs)
        self._plan = plan

    def start_episode(self, state):
        super().start_episode(state)
        self.plan = iter(self._plan)
        # self.model = Model(self.env)

    def act(self, state):
        return next(self.plan)

def heuristic(env, obs):
    row, col = obs
    g_row, g_col = env.goal
    return (abs(row - g_row) + abs(col - g_col))


from models import BayesianRegression

class MetaBestFirstSearchPolicy(Policy):
    """Chooses computations in a MetaBestFirstSearchEnv."""
    def __init__(self, theta=None, n_iter=1):
        FEATURES = 2
        super().__init__()
        self.theta = theta
        if theta is None:
            self.n_iter = n_iter
            self.memory_length = 1000
            self.V = BayesianRegression(FEATURES)
            self.history = defaultdict(lambda: deque(maxlen=self.memory_length))

    def phi(self, node):
        # empty = not node.path
        reward_so_far = node.reward
        distance = heuristic(self.env.env, node.state)
        x = np.r_[reward_so_far, distance]
        return x

    @curry
    def eval_node(self, node, noisy):
        if node is None:
            return -np.inf
        elif self.theta:
            return self.theta @ self.phi(node)
        else:
            v, var = self.V.predict(self.phi(node), return_var=True)
            if noisy and self.i_episode <= 0:
                return v + np.random.randn() * var
            else:
                return v


    def finish_episode(self, trace):
        if self.theta is None:
            w, var = self.V.weights.get_moments()
            self.save('weights', w)
            self.save('weights_var', var)
            X = [self.phi(node) for node in trace['actions'][:-1]]
            y = list(reversed(np.cumsum(list(reversed(trace['rewards'])))))[:-1]

            self.history['X'].extend(X)
            self.history['y'].extend(y)

            X = np.stack(self.history['X'])
            y = np.array(self.history['y'])#.reshape(-1, 1)
            self.V.fit(X, y)

            # self.save('w', self.V.w)
            # X = np.array([self.phi(node) for node in trace['actions'][:-1]])
            # y = list(reversed(np.cumsum(list(reversed(trace['rewards'])))))
            # y = np.array(y[:-1]).reshape(-1, 1)
            # self.save('X', X)
            # self.save('y', y)
            # self.V.update(X, y)

            # h = self.model.fit(X, y, batch_size=len(X), epochs=1, verbose=0)
            # loss = h.history['loss']
            # from keras import backend as K
            # with K.get_session().as_default():
            #     self.save('weights', self.model.weights[0].eval())
            # self.save('loss', loss)
            

    def act(self, state):
        frontier, reward_to_state, best_done = state
        # print('frontier', frontier)
        # self.save('frontier', [n[1].state for n in frontier])
        self.ep_trace['frontier'].append([n[1].state for n in frontier])

        if best_done:
            value = self.eval_node(best_done, noisy=False)
            best = all(value > self.eval_node(n, noisy=False) 
                       for v, n in frontier)
            if best:
                return 'TERM'

        if frontier:
            return frontier.pop()
        else:
            print('NO FRONTIER')
            assert 0, 'no frontier'



class MouselabEnv(gym.Env):
    """MetaMDP for a tree with a discrete unobserved reward function."""
    metadata = {'render.modes': ['human', 'array']}
    term_state = None
    def __init__(self, branch=2, height=2, reward=None, cost=0, ground_truth=None):
        self.branch = branch
        if hasattr(self.branch, '__len__'):
            self.height = len(self.branch)
        else:
            self.height = height
            self.branch = [self.branch] * self.height

        self.cost = - abs(cost)
        self.reward = reward if reward is not None else Normal(1, 1)
        self.ground_truth = ground_truth

        self.tree = self._build_tree()
        self.init = (0,) + (self.reward,) * (len(self.tree) - 1)
        self.term_action = len(self.tree)
        self.reset()

        self.action_space = spaces.Discrete(len(self.tree) + 1)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=len(self.tree))

    def _reset(self):
        self._state = self.init
        return self.features(self._state)

    def _step(self, action):
        if self._state is self.term_state:
            print('BAD')
            return None, 0, True, {}
        if action == self.term_action:
            # self._state = self.term_state
            reward = self.term_reward().sample()
            done = True
        elif self._state[action] is not self.reward:  # already observed
            reward = 0
            done = False
        else:  # observe a new node
            self._state = self._observe(action)
            reward = self.cost
            done = False
        return self.features(self._state), reward, done, {}

    def _observe(self, action):
        if self.ground_truth is not None:
            result = self.ground_truth[action]
        else:
            result = self._state[action].sample()
        s = list(self._state)
        s[action] = result
        return tuple(s)

    def features(self, state=None):
        state = state if state is not None else self._state
        if state is None:
            return np.full(len(self.tree), np.nan)
        # Is each node observed?
        return np.array([1. if hasattr(x, 'sample') else 0.
                         for x in state])
    
    def action_features(self, action, state=None):
        state = state if state is not None else self._state
        if state is None:
            assert 0

        if action == self.term_action:
            tr_mu, tr_sigma = norm.fit(self.term_reward.sample(10000))
            return np.r_[0, 0, 0, 0, 0, tr_mu, tr_sigma]

        nq_mu, nq_sigma = norm.fit(self.node_quality(action).sample(10000))
        nqpi_mu, nqpi_sigma = norm.fit(self.node_quality(action).sample(10000))
        return np.r_[1, nq_mu, nq_sigma, nqpi_mu, nqpi_sigma, 0, 0]

    def term_reward(self, state=None):
        state = state if state is not None else self._state
        # state = self._state
        assert state is not None
        return self.node_value(0, state)

    def node_value(self, node, state=None):
        """A distribution over total rewards after the given node."""
        state = state if state is not None else self._state
        return max((self.node_value(n1, state) + state[n1]
                    for n1 in self.tree[node]), 
                   default=PointMass(0), key=expectation)

    def node_value_to(self, node, state=None):
        """A distribution over rewards up to and including the given node."""
        state = state if state is not None else self._state
        start_value = PointMass(0)
        return sum((state[n] for n in self.path_to(node)), start_value)

    def node_quality(self, node, state=None):
        """A distribution of total expected rewards if this node is visited."""
        state = state if state is not None else self._state
        return self.node_value_to(node, state) + self.node_value(node, state)

    def node_value_omniscient(self, node, state=None):
        """A distribution of the value of a node given knowledge of all rewards."""
        state = state if state is not None else self._state
        children = self.tree[node]
        if children:
            return dmax(self.node_value_omniscient(c) + state[c] for c in children)
        else:
            return PointMass(0)

    def node_quality_omniscient(self, node, state=None):
        """A distribution of total expected rewards if this node is visited."""
        state = state if state is not None else self._state
        return self.node_value_to(node, state) + self.node_value_omniscient(node, state)

    def node_quality_observed(self, node, state=None):
        """A distribution of total expected rewards if this node were observed."""
        state = state if state is not None else self._state
        return self.node_value_to(node, state) + self.node_value_omniscient(node, state)

    def path_to(self, node, start=0):
        path = [start]
        if node == start:
            return path
        for _ in range(self.height + 1):
            children = self.tree[path[-1]]
            for i, child in enumerate(children):
                if child == node:
                    path.append(node)
                    return path
                if child > node:
                    path.append(children[i-1])
                    break
            else:
                path.append(child)
        assert False

    def all_paths(self, start=0):
        def rec(path):
            children = self.tree[path[-1]]
            if children:
                for child in children:
                    yield from rec(path + [child])
            else:
                yield path

        return rec([start])

    def subtree(self, state, n):
        """Returns the substree of the belief state with root n."""
        assert 0
        if not self.tree[n]:  # leaf node
            return state[n]
        c1, c2 = self.tree[n]
        return tuple(state[i] for i in range(n, 2 * c2 - c1))

    def _build_tree(self):
        # num_node = np.cumsum(self.branch).sum() + 1
        def nodes_per_layer():
            n = 1
            yield n
            for b in self.branch:
                n *= b
                yield n

        num_node = sum(nodes_per_layer())
        T = [[] for _ in range(num_node)]  # T[i] = [c1, c2, ...] or [] if i is terminal

        ids = it.count(0)
        def expand(i, d):
            if d == self.height:
                return
            for _ in range(self.branch[d]):
                next_i = next(ids)
                T[i].append(next_i)
                expand(next_i, d+1)

        expand(next(ids), 0)
        return T

    # def _build_tree(self):
    #     """Constructs the transition object-level MDP."""
    #     num_node = self.branch ** (self.height+1) - 1
    #     T = [[] for _ in range(num_node)]  # T[i] = [c1, c2] or [] if i is terminal
    #     ids = it.count(0)
    #     def expand(i, d):
    #         if d == self.height:
    #             return
    #         for _ in range(self.branch):
    #             next_i = next(ids)
    #             T[i].append(next_i)
    #             expand(next_i, d+1)

    #     expand(next(ids), 0)
    #     return T

    def _render(self, mode='notebook', close=False):
        from graphviz import Digraph
        from IPython.display import display
        import matplotlib as mpl
        from matplotlib.colors import rgb2hex
        if close:
            return
        
        vmin = -2
        vmax = 2
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = mpl.cm.get_cmap('RdYlGn')
        colormap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        colormap.set_array(np.array([vmin, vmax]))

        def color(val):
            if val > 0:
                return '#8EBF87'
            else:
                return '#F7BDC4'
            # return '#9999ee'
            # return rgb2hex(colormap.to_rgba(val))
        
        # COLOR = {None: 'grey', np.inf: 'grey', 1: '#dd4444', 0: '#5555ee'}
        dot = Digraph()
        for x, ys in enumerate(self.tree):
            r = self._state[x]
            observed = not hasattr(self._state[x], 'sample')
            c = color(r) if observed else 'grey'
            l = str(round(r, 2)) if observed else '?'
            dot.node(str(x), label=l, style='filled', color=c)
            for y in ys:
                dot.edge(str(x), str(y))
        display(dot)
