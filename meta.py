from collections import namedtuple, defaultdict, deque, Counter
import numpy as np
from utils import PriorityQueue
from agents import Agent, Model
import gym
from policies import Policy
from toolz import memoize, curry
import itertools as it
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
            if node1.reward <= reward_to_state[s1]:
                continue  # cannot be better than an existing node
            reward_to_state[s1] = node1.reward
            if done:
                # return node1  # ASSUMPTION: only one path to goal
                best_done = min((best_done, node1), key=self.eval_node(noisy=False))
            else:
                frontier.push(node1)

        self._state = self.State(frontier, reward_to_state, best_done)
        return self._state


class FixedPlanPolicy(Policy):
    """A policy that blindly executes a fixed sequence of actions."""
    Node = namedtuple('Node', ('state', 'path', 'reward', 'done'))
    def __init__(self, plan, **kwargs):
        super().__init__(**kwargs)
        self.plan = iter(plan)

    def start_episode(self, state):
        super().start_episode(state)
        self.history = Counter()
        # self.model = Model(self.env)

    def act(self, state):
        self.history[state] += 1
        return next(self.plan)

def heuristic(env, obs):
    row, col = obs
    g_row, g_col = env.goal
    return (abs(row - g_row) + abs(col - g_col))


class MetaBestFirstSearchPolicy(Policy):
    """Chooses computations in a MetaBestFirstSearchEnv."""
    def __init__(self, theta=None):
        FEATURES = 2
        super().__init__()
        self.theta = theta
        if not theta:
            from keras.layers import Dense
            from keras.models import Sequential
            self.model = Sequential([
                Dense(1, input_dim=FEATURES, activation='linear'),
            ])
            self.model.compile(optimizer='adam', loss='mse')


    def phi(self, node):
        # empty = not node.path
        obs = node.state
        reward_so_far = node.reward
        distance = heuristic(self.env.env, obs)
        x = np.r_[reward_so_far, distance]
        return x

    @curry
    def eval_node(self, node, noisy):
        if node is None:
            return np.inf
        if self.theta:
            return self.theta @ self.phi(node)
        else:
            v = self.model.predict(self.phi(node).reshape(1, -1))[0]
            noise = np.random.rand() * (.95 ** self.i_episode) if noisy else 0
            return - (v + noise)


    def finish_episode(self, trace):
        if not self.theta:
            X = np.array([self.phi(s) for s in trace['actions'][:-1]])
            self.save('X', X)
            y = list(reversed(np.cumsum(list(reversed(trace['rewards'])))))
            y = np.array(y[:-1]).reshape(-1, 1)
            self.save('y', y)
            h = self.model.fit(X, y, batch_size=len(X), epochs=1, verbose=0)
            loss = h.history['loss']
            from keras import backend as K
            with K.get_session().as_default():
                self.save('weights', self.model.weights[0].eval())
            self.save('loss', loss)
            

    def act(self, state):
        frontier, reward_to_state, best_done = state
        # print('frontier', frontier)
        self.save('frontier', [n[1].state for n in frontier])

        if best_done:
            return 'TERM'
        elif frontier:
            return frontier.pop()
        else:
            assert 0, 'no frontier'



class MouselabEnv(gym.Env):
    """MetaMDP for a tree with a discrete unobserved reward function."""
    metadata = {'render.modes': ['human', 'array']}
    term_state = None
    def __init__(self, branch=2, height=2, reward=None, cost=0):
        self.branch = branch
        self.height = height
        self.cost = - abs(cost)
        self.reward = reward if reward is not None else Normal(1, 1)
        self.expected_reward = expectation(self.reward)

        self.tree = self._build_tree()
        self.init = (self.reward,) * len(self.tree)
        self.term_action = len(self.tree)
        self.reset()

        self.action_space = gym.spaces.Discrete(len(self.tree) + 1)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=len(self.tree))

    def _reset(self):
        self._state = self.init
        return self._observe(self._state)

    def _step(self, action):
        if self._state is self.term_state:
            print('BAD')
            return None, 0, True, {}
        assert self._state is not None, '2'
        if action == self.term_action:
            reward = self.term_reward().sample()
            self._state = self.term_state
            done = True
        elif self._state[action] is not self.reward:  # already observed
            reward = self.cost
            # reward = -1
            done = False
        else:  # observe a new node
            s = list(self._state)
            s[action] = s[action].sample()
            self._state = tuple(s)
            reward = self.cost
            done = False
        return self._observe(self._state), reward, done, {}

    def _observe(self, state=None):
        if state is None:
            return np.full(len(self.tree), np.nan)
        state = state if state is not None else self._state
        # Is each node observed?
        return np.array([1. if not hasattr(x, 'sample') else 0.
                         for x in state])

    def term_reward(self, state=None):
        state = state if state is not None else self._state
        # state = self._state
        assert state is not None
        return self.tree_V(0, state)

    def encode(self, state):
        pass

    def observe(self, node, state=None):
        state = state if state is not None else self._state
        s = list(state)
        s[node]

    def tree_V(self, s=0, state=None):
        state = state if state is not None else self._state
        # includes the reward attained at state s (it's not really a value function)
        r = state[s]
        future_reward = max((self.tree_V(s1, state) for s1 in self.tree[s]), 
                            default=Discrete((0,)), key=expectation)
        return future_reward + r

    def subtree(self, state, n):
        """Returns the substree of the belief state with root n."""
        if not self.tree[n]:  # leaf node
            return state[n]
        c1, c2 = self.tree[n]
        return tuple(state[i] for i in range(n, 2 * c2 - c1))

    def _build_tree(self):
        """Constructs the transition object-level MDP."""
        num_node = self.branch ** (self.height+1) - 1
        T = [[] for _ in range(num_node)]  # T[i] = [c1, c2] or [] if i is terminal
        ids = it.count(0)
        def expand(i, d):
            if d == self.height:
                return
            for _ in range(self.branch):
                next_i = next(ids)
                T[i].append(next_i)
                expand(next_i, d+1)

        expand(next(ids), 0)
        return T

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
            # return '#9999ee'
            return rgb2hex(colormap.to_rgba(val))
        
        # COLOR = {None: 'grey', np.inf: 'grey', 1: '#dd4444', 0: '#5555ee'}
        dot = Digraph()
        for x, ys in enumerate(self.tree):
            r = self._state[x]
            observed = not hasattr(self._state[x], 'sample')
            c = color(r) if observed else 'grey'
            l = str(r) if observed else '?'
            dot.node(str(x), label=l, style='filled', color=c)
            for y in ys:
                dot.edge(str(x), str(y))
        display(dot)


def expectation(val):
    if hasattr(val, 'expectation'):
        return val.expectation()
    else:
        return val

def sample(val):
    if hasattr(val, 'sample'):
        return val.sample()
    else:
        return val

def cross(d1, d2, f):
    outcomes = Counter()
    for ((o1, p1), (o2, p2)) in it.product(d1, d2):
        outcomes[f(o1, o2)] += p1 * p2

    return Discrete(outcomes.keys(), outcomes.values())


class Normal(object):
    """Normal distribution."""
    def __init__(self, mu, sigma):
        super().__init__()
        self.mu = mu
        self.sigma = sigma

    def __repr__(self):
        return 'Norm'

    def __add__(self, other):
        if isinstance(other, Normal):
            return Normal(self.mu + other.mu, self.sigma + other.sigma)
        else:
            return Normal(self.mu + other, self.sigma)

    def expectation(self):
        return self.mu

    def sample(self):
        return self.mu + self.sigma * np.random.randn()


class Discrete(object):
    """Discrete distribution."""
    def __init__(self, vs, ps=None):
        super().__init__()
        self.vs = tuple(vs)
        if ps is None:
            self.ps = tuple(1/len(vs) for _ in range(len(vs)))
        else:
            self.ps = tuple(ps)

    def __repr__(self):
        return 'Cat'

    def __iter__(self):
        return zip(self.vs, self.ps)

    def __len__(self):
        return len(self.ps)

    def __add__(self, other):
        if isinstance(other, Discrete):
            return cross(self, other, lambda s, o: s + o)
        else:
            return self.apply(lambda v: v + other)

    def apply(self, f):
        vs = tuple(f(v) for v in self.vs)
        return Discrete(vs, self.ps)

    def expectation(self):
        return sum(p * v for p, v in zip(self.ps, self.vs))

    def sample(self):
        return np.random.choice(self.vs, p=self.ps)
      
      

