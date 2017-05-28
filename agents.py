"""Agents that operate in discrete fully observable environments."""

from collections import namedtuple, defaultdict, Counter, deque
import numpy as np
from abc import ABC, abstractmethod
import utils
import time
np.set_printoptions(precision=3, linewidth=200)

from copy import deepcopy

# ========================== #
# ========= Agents ========= #
# ========================== #

class Agent(ABC):
    """An agent that can run on  discrete openai gym environments.

    All agents inherit from this abstract base class (which itself cannot be
    instantiated). A class implementing Agent must override act. Any learning
    algorithm (e.g. Sarsa) will also implement update."""
    def __init__(self, env, discount=0.99):
        self.env = env
        self.i_episode = 0
        self.discount = discount

        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n

    @abstractmethod  # all subclases must implement this method
    def act(self, state):
        """Returns an action to take in the given state.

        A state is an int between 0 and self.n_states
        An action is an int between 0 and self.n_actions.
        """
        pass

    def update(self, state, action, new_state, reward, done):
        """Learn from the results of taking action in state.

        state: state in which action was taken.
        action: action taken.
        new_state: the state reached after taking action.
        reward: reward received after taking action.
        done: True if the episode is complete.
        """
        pass

    def start_episode(self, state):
        """This function is run once when an episode begins, starting at state.

        This can be used to e.g. to initialize episode-specific memory as necessary
        for n-step TD learning."""
        pass

    def run_episode(self, render=False, max_steps=1000, interact=False,
                    verbose=False):
        """Runs a single episode, returns a complete trace of the episode."""
        self.log = print if verbose else (lambda *args: None)


        if interact:
            render = 'human'
            last_cmd = ''
        trace = {
            'i_episode': self.i_episode,
            'states': [],
            'actions': [],
            'rewards': [],
            'finished': False,
            'return': None
        }
        new_state = self.env.reset()
        trace['_state'] = self.env._state
        self.start_episode(new_state)
        done = False
        for i_step in range(max_steps):
            state = new_state

            self._render(render)

            if interact:
                cmd = input('> ') or last_cmd
                if cmd == 'break':
                    import ipdb; ipdb.set_trace()
                if cmd == 'exit':
                    exit()

            action = self.act(state)
            new_state, reward, done, info = self.env.step(action)
            self.update(state, action, new_state, reward, done)
            
            trace['states'].append(state)
            trace['actions'].append(action)
            trace['rewards'].append(reward)

            if done:
                trace['finished'] = True
                self._render(render)
                break

        trace['states'].append(new_state)  # final state
        self.i_episode += 1
        trace['return'] = sum(trace['rewards'])   # TODO discounting
        return trace

    def run_many(self, n_episodes, track=(), **kwargs):
        """Runs several episodes, returns a summary of results."""
        data = defaultdict(list)
        for _ in range(n_episodes):
            trace = self.run_episode(**kwargs)
            data['i_episode'].append(trace['i_episode'])
            data['n_steps'].append(len(trace['states']))
            data['return'].append(trace['return'])
            data['finished'].append(trace['finished'])
            for k, v in self.trace().items():
                data[k].append(v)

        return data

    def _render(self, mode):
        if mode == 'step':
            x = input('> ')
            while x:
                print(eval(x))
                x = input('> ')
            utils.clear_screen()
            self.env.render()
        elif mode == 'clear':
            utils.clear_screen()
            self.env.render()
        elif mode == 'auto':
            time.sleep(.3)
            utils.clear_screen()
            self.env.render()
        elif mode:
            self.env.render(mode=mode)


class TDLambdaV(object):
    """Learns a linear value function with TD lambda."""
    def __init__(self, env, learn_rate=.01, trace_decay=0, decay=1, memory=10):
        self.env = env
        self.learn_rate = learn_rate
        self.shape = len(self.features(env.reset()))
        self.trace_decay = trace_decay
        self.decay = decay
        self.theta = np.zeros(self.shape)
        # self.memory = deque(maxlen=memory)
        self.trace = np.zeros(self.shape)

    def features(self, s):
        if not isinstance(s, int):
            return np.r_[1, s]
        x = [0] * self.env.nS
        x[s] = 1
        return x
        # return self.env.decode(s)

    def update(self, s, v):
        # self.learn_rate *= .999
        x = self.features(s)
        vhat = x @ self.theta
        error = v - vhat
        self.trace = self.trace_decay * self.trace + x
        # if np.random.rand() < .01:
        #     print('------')
        #     print(error)
        #     print(self.trace)
        #     print(self.theta)
        self.theta += self.learn_rate * error * self.trace
        self.theta *= self.decay
        # self.memory.append(self.theta.copy())

    def predict(self, s, ucb=False):
        x = self.features(s)
        return x @ self.theta

    def to_array(self):
        return np.array([self.predict(s) for s in range(self.env.nS)])

    __call__ = predict



class PlanAgent(Agent):
    """An Agent with a plan."""
    def __init__(self, env, replan=False, **kwargs):
        super().__init__(env, **kwargs)
        self.replan = replan
        self.plan = iter([])

    def act(self, state):
        try:
            if self.replan:
                raise StopIteration()
            else:
                return next(self.plan)
        except StopIteration:
            self.plan = iter(self.make_plan(state))
            return next(self.plan)




class SearchAgent(PlanAgent):
    """Searches for the maximum reward path using a model."""

    def __init__(self, env, depth=None, **kwargs):
        super().__init__(env, **kwargs)
        if depth is None:
            depth = -1  # infinite depth
        self.depth = depth

        # if model is None:
        #     model = TrueModel(env)
        self.model = None
        self.last_state = None
        self.V = TDLambdaV(env)

    def start_episode(self, state):
        self.explored = set()
        self.model = deepcopy(self.env)

    def trace(self):
        return {
            'theta': self.V.theta.copy(),
            'berries': self.env._observe()[-1]
        }

    def act(self, s0):
        # return self.env.action_space.sample()
        self.explored.add(s0)
        return super().act(s0)

    def update(self, s0, a, s1, r, done):
        target = r + self.discount * self.V(s1)
        self.V.update(s0, target)

    def make_plan(self, state):

        Node = namedtuple('Node', ('state', 'path', 'value'))
        model = self.model

        def options(state):
            for a in range(self.n_actions):
                model._state = state
                obs, r, done, info = model.step(a)
                yield a, model._state, r, done


        def expand(node):
            s0, p0, v0 = node
            for a, s1, r, done in options(s0):
                p1, v1 = p0 + [a], v0 + r
                # print(len(p1))
                if done:
                    yield Node(s1, p1, v1)
                elif len(p1) > self.depth:
                    noise = np.random.rand() * .98 ** self.i_episode
                    value = self.V(self.env._observe(s1)) + noise
                    yield Node(s1, p1, v1 + value)
                else:
                    yield from expand(Node(s1, p1, v1))


        def eval_node(node):
            exploring = 0
            return (node.value, exploring)

        node = Node(self.env._state, [], 0)
        return max(expand(node), key=eval_node).path




class LinearQ(object):
    """Learns a linear Q function by SGD."""
    def __init__(self, shape, learn_rate=.1):
        self.shape = shape
        self.learn_rate = learn_rate
        self.theta = np.random.random(self.shape)

    def update(self, x, y):
        yhat = x @ self.theta
        error = y - yhat
        self.theta += self.learn_rate * np.outer(x, error)

    def predict(self, x):
        return x @ self.theta

    __call__ = predict


class QLearningAgent(Agent):
    """Learns expected values of taking an action in a state."""
    def __init__(self, env, learn_rate=.85, discount=.99, epsilon=.5, anneal=.99,
                 exploration='epsilon'):
        super().__init__(env, discount=discount)
        self.epsilon = epsilon
        self.anneal = anneal
        self.exploration = exploration
        shape = (len(env.decode(0)), env.nA)
        self.Q = LinearQ(shape, learn_rate)

    def act(self, state):
        if self.exploration:
            epsilon = self.epsilon * self.anneal ** self.i_episode
            if self.exploration == 'epsilon' and np.random.rand() < epsilon:
                return np.random.randint(self.Q.shape[1])

            elif self.exploration == 'noise':
                noise = np.random.randn(self.Q.shape[1]) * epsilon
                return np.argmax(self.Q(state) + noise)

        noise = np.random.randn(self.Q.shape[1]) * .001
        return np.argmax(self.Q(state) + noise)

    def update(self, s0, a, s1, r, done):
        target = self.Q(s0)
        target[a] = r + self.discount * np.max(self.Q(s1))
        self.Q.update(s0, target)


class Model(object):
    """Learned model of an MDP."""
    Node = namedtuple('Node', ['p','s0', 'a', 's1', 'r', 'done'])

    def __init__(self, env):
        # (s, a) -> [total_count, outcome_counts]
        self.env = deepcopy(env)
        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n
        self.counts = defaultdict(lambda: [0, Counter()])

    def options(self, s):
        """Yields possible outcomes of taking action a in state s.

        Outcome is (prob, s1, r, done).
        """
        for a in range(self.n_actions):
            env.state = s

        total_count, outcomes = self.counts[s, a]
        for (s1, r, done), count in outcomes.items():
            yield (count / total_count, s1, r, done)

    def expected_reward(self, s0, a, s1=None):
        return sum(p * r for (p, s1_, r, done) in self.results(s0, a)
                   if s1 is None or s1 == s1_)

    def update(self, s0, a, s1, r, done):
        self.counts[s0, a][0] += 1  # total count
        self.counts[s0, a][1][s1, r, done] += 1  # outcome count

    def paths(self, state, depth=-1, cycles=False):
        """Yields all paths to a final state or a state `depth` steps away."""

        def expand(path, explored, depth):
            # TODO explored!
            s0 = path[-1].s1 if path else state  # initial call
            for a in range(self.n_actions):
                for (p, s1, r, done) in self.results(s0, a):
                    if p and (cycles or s1 not in explored):
                        new_path = path + [self.Node(p, s0, a, s1, r, done)]
                        if done or depth == 1:
                            yield new_path
                        else: 
                            yield from expand(new_path, explored | {s1}, depth-1)

        yield from expand([], {state}, depth)


class TrueModel(Model):
    """Accurate model of a DiscreteEnv."""
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)

    def results(self, state, action):
        yield from self.env.P[state][action]

    def update(*args):
        pass


class ModelBasedAgent(Agent):
    """Agent that builds a model of the MDP."""
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        self.model = Model(env)
        self.Q = np.zeros((self.model.n_states, self.model.n_actions))

    def act(self, state):
        noise = np.random.randn(self.Q.shape[1]) / (self.i_episode + 1)
        return np.argmax(self.Q[state] + noise)  # a = action

    def update(self, s0, a, s1, r, done):
        self.model.update(s0, a, s1, r, done)
        self.update_policy(s0, a)

    def update_policy(self, s, a):
        Q, V = self.Q, self.V
        expected_future_reward = sum(p * V(s1) for p, s1, r, done in self.model.results(s, a))
        Q[s, a] = self.model.expected_reward(s, a) + self.discount * expected_future_reward

    def V(self, s):
        """The value of taking the best possible action from state s."""
        return self.Q[s].max()


class PrioritizedSweeping(ModelBasedAgent):
    """Learns by replaying past experience.

    https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume4/kaelbling96a-html/node29.html
    """
    def __init__(self, env, n_simulate=0, **kwargs):
        super().__init__(env, **kwargs)
        self.n_simulate = n_simulate
        self.value_change = np.zeros(self.n_states)
        self.predecessors = defaultdict(set)

    def update(self, s0, a, s1, r, done):
        super().update(s0, a, s1, r, done)

        self.predecessors[s1].add(s0)
        # Update Q by simulation.
        for _ in range(self.n_simulate):
            s = self.value_change.argmax()
            if self.value_change[s] == 0:
                break  # no new information to propogate
            self.value_change[s] = 0  # reset
            self.update_policy(s, self.act(s))

    def update_policy(self, s0, a):
        # Track changes to prioritize simulations.
        old_val = self.V(s0)
        super().update_policy(s0, a)
        change = abs(self.V(s0) - old_val)
        if change:
            self.log('change', s0, a, change)
        for s_pred in self.predecessors[s0]:
            self.value_change[s_pred] = max(self.value_change[s_pred], change)  # TODO weight by transition prob


def value_iteration(env, discount=.999, epsilon=0.001, max_iters=100000):
    """Returns the optimal value table for env."""
    V1 = np.zeros(env.observation_space.n)

    def value(result):
        # [(0.5, 0, 0, False), (0.5, 0, 0, False)] -> float
        return sum(p * (r + discount * V[s1])
                   for p, s1, r, _ in result)

    for i in range(1, max_iters+1):
        V = V1.copy()
        delta = 0
        for state, actions in env.P.items():

            # Example actions object. Keys are actions, values are
            # lists of (prob, next_state, reward, done).
            # {0: [(1.0, 1, 0, False)],
            #  1: [(1.0, 2, 0, False)],
            #  2: [(1.0, 3, -1, False)],
            #  3: [(1.0, 0, 4, False)]}

            results = actions.values()
            V1[state] = max(map(value, results))
            delta = max(delta, abs(V1[state] - V[state]))

        if delta < epsilon * (1 - discount) / discount:
            return V

    print('NOT CONVERGED')
    return V



def v_to_q(env, V):
    
    def rval(result):
        p, s1, r, _ = result
        return p * (r + V[s1])
    
    def qval(s, a):
        return sum(rval(result) for result in env.P[s][0])
    
    Q = np.array([[qval(s, a) for a in range(env.n_actions)]
                  for s in range(env.n_states)])
    return Q







