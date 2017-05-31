"""Agents that operate in discrete fully observable environments."""

from collections import namedtuple, defaultdict, Counter, deque
import itertools as it
import numpy as np
from abc import ABC, abstractmethod
from utils import clear_screen, PriorityQueue
import time
np.set_printoptions(precision=3, linewidth=200)

from tqdm import tqdm, trange, tnrange
from copy import deepcopy
from toolz.curried import *
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
        """This function is run when an episode begins, starting at state.

        This can be used to e.g. to initialize episode-specific memory as necessary
        for n-step TD learning."""
        pass

    def finish_episode(self):
        """This function is run when an episode ends."""
        return


    def run_episode(self, render=False, max_steps=1000, interact=False,
                    verbose=False):
        """Runs a single episode, returns a complete trace of the episode."""
        self.log = print if verbose else (lambda *args: None)


        if interact:
            render = 'human'
            last_cmd = ''
        trace = self._trace = defaultdict(list)
        trace.update({
                    'i_episode': self.i_episode,
                    'states': [],
                    'actions': [],
                    'rewards': [],
                    'finished': False,
                    'return': None
                })

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
        trace['return'] = sum(trace['rewards'])   # TODO discounting
        self.finish_episode()
        self.i_episode += 1
        return trace


    def run_many(self, n_episodes, track=(), **kwargs):
        """Runs several episodes, returns a summary of results."""
        data = defaultdict(list)
        for _ in tnrange(n_episodes):
            trace = self.run_episode(**kwargs)
            data['n_steps'].append(len(trace.pop('states')))
            # data['i_episode'].append(trace.pop('i_episode'))
            # data['return'].append(trace.pop('return'))
            # data['finished'].append(trace.pop('finished'))
            trace.pop('actions')
            trace.pop('rewards')
            for k, v in trace.items():
                data[k].append(v)

        return data

    def _render(self, mode):
        if mode == 'step':
            x = input('> ')
            while x:
                print(eval(x))
                x = input('> ')
            clear_screen()
            self.env.render()
        elif mode == 'clear':
            clear_screen()
            self.env.render()
        elif mode == 'auto':
            time.sleep(.4)
            clear_screen()
            self.env.render()
        elif mode:
            self.env.render(mode=mode)



class ValueFunction(object):
    """Learns a linear value function with TD lambda."""
    def __init__(self, env, discount=1, learn_rate=.01, decay=1):
        self.env = env
        self.learn_rate = learn_rate
        self.discount = discount
        self.shape = len(self.features(env.reset()))
        self.decay = decay

    def features(self, s):
        if not isinstance(s, int):
            return np.r_[1, s]
        x = [0] * self.env.nS
        x[s] = 1
        return x

    def update(self, s0, a, s1, r, done):
        return

    def finish_episode(self, trace):
        pass

    def predict(self, s):
        x = self.features(s)
        return x @ self.theta

    def start_episode(self, i_ep):
        return

    def to_array(self):
        return np.array([self.predict(s) for s in range(self.env.nS)])

def interactions(x):
    return [a * b for a, b in it.combinations(x, 2)]

class FixedV(ValueFunction):
    """User-specified value function."""
    def __init__(self, env, theta):
        super().__init__(env)
        self.theta = np.array(theta)
      

class TDLambdaV(ValueFunction):
    """Learns a linear value function with TD lambda."""
    def __init__(self, env, trace_decay=0, **kwargs):
        super().__init__(env, **kwargs)
        self.trace_decay = trace_decay
        self.trace = np.zeros(self.shape)
        self.theta = np.zeros(self.shape)
        self.theta_update = np.zeros(self.shape)

    def start_episode(self, i_ep):
        self.theta = self.theta_update.copy()

    def update(self, s0, a, s1, r, done):
        target = r + self.discount * self.predict(s1)
        x = self.features(s0)
        prediction = x @ self.theta
        error = target - prediction
        # print(x)
        # import time; time.sleep(0.1)
        self.trace = self.trace_decay * self.trace + x
        self.theta_update += self.learn_rate * error * self.trace
        self.theta_update *= self.decay


    def features(self, s):
        # return np.r_[1, s, interactions(s)]
        return np.r_[1, s]

    # def update(self, s0, a, s1, r, done):
    #     target = r + self.discount * self.predict(s1)
    #     x = self.features(s0)
    #     prediction = x @ self.theta
    #     error = target - prediction
    #     self.trace = self.trace_decay * self.trace + x
    #     self.theta += self.learn_rate * error * self.trace
    #     self.theta *= self.decay

def scratch():
    agent = Agent()
    V = TDLambdaV()
    agent.register(V)
    policy = Search(V)
    





from sklearn.linear_model import SGDRegressor
class MonteCarloV(ValueFunction):
    """Learns a linear value function with every-step Monte Carlo."""
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        self.model = SGDRegressor()

    def finish_episode(self, trace):
        X = np.array([self.features(s) for s in trace['states']])
        y = list(reversed(np.cumsum(list(reversed(trace['rewards'])))))
        y.append(0)  # value of final state
        self.model.partial_fit(X, y)

    def predict(self, s):
        if self.model.coef_ is not None:
            return self.model.predict(self.features(s).reshape(1, -1))[0]
        else:
            return 0

    @property
    def theta(self):
        return self.model.coef_

        
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
        self.V = TDLambdaV(env, trace_decay=0, decay=.99, learn_rate=.01)
        # self.V = FixedV(env, [0, 0, 0, 1])

    def start_episode(self, state):
        self.history = Counter()
        self.model = Model(self.env)
        self.V.start_episode(self.i_episode)

    def finish_episode(self):
        self.V.finish_episode(self._trace)
        self._trace.update({
            'theta': self.V.theta.copy(),
            'berries': self.env._observe()[-1],
        })
        t = self._trace
        print(len(t['paths']), round(t['return'], 2))

    def act(self, s0):
        # return self.env.action_space.sample()
        self.history[s0] += 1
        return super().act(s0)


    def update(self, s0, a, s1, r, done):
        self.V.update(s0, a, s1, r, done)

    def make_plan(self, state, expansions=5000):

        Node = namedtuple('Node', ('state', 'path', 'reward', 'done'))
        @curry
        def eval_node(node, noisy=1):
            noise = np.random.rand() * 5 * .99 ** self.i_episode if noisy else 0
            value = 0 if node.done else self.V.predict(self.env._observe(node.state))

            boredom = - 0.1 * self.history[self.env._observe(node.state)]
            score = node.reward + value + noise + boredom
            return - score

        start = Node(self.env._state, [], 0, False)
        frontier = PriorityQueue(key=eval_node)
        frontier.push(start)
        completed = []
        def expand(node):
            s0, p0, v0, _ = node
            for a, s1, r, done in self.model.options(s0):
                node1 = Node(s1, p0 + [a], v0 + r, done)
                if done:
                    completed.append(node1)
                else:
                    frontier.push(node1)
                    
        for _ in range(expansions):
            if frontier:
                expand(frontier.pop())
            else:
                break


        choices = concat([completed, map(get(1), take(100, frontier))])
        plan = min(choices, key=eval_node(noisy=True))
        # print(len(plan.path), -round(eval_node(plan, noisy=False), 2), plan.done)
        self._trace['paths'].append(plan.path)
        return plan.path


class Model(object):
    """Simulated environment"""
    def __init__(self, env):
        self.env = deepcopy(env)
      
    def options(self, state):
        for a in range(self.env.action_space.n):
            self.env._state = state
            obs, r, done, info = self.env.step(a)
            yield a, self.env._state, r, done



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



