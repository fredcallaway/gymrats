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

class RegistrationError(Exception): pass

class Agent(ABC):
    """An agent that can run openai gym environments."""
    def __init__(self):
        self.env = None
        self.policy = None
        self.ep_trace = None
        self.value_functions = []
        self.i_episode = 0

    @property
    def n_actions(self):
        return self.env.action_space.n

    def register(self, obj):
        """Attaches a component or env to this agent."""
        if hasattr(obj, 'step'):  # gym Env
            self.env = obj
        elif hasattr(obj, 'act'):
            self.policy = obj
            obj.attach(self)
        elif hasattr(obj, 'predict'):
            self.value_functions.append(obj)
            obj.attach(self)
        else:
            raise ValueError('Cannot register {}'.format(obj))

    def run_episode(self, render=False, max_steps=1000, interact=False,
                    verbose=False, reset=True):
        """Runs a single episode, returns a complete trace of the episode."""
        if not self.env:
            raise RegistrationError('No environment registered.')
        if not self.policy:
            raise RegistrationError('No policy registered.')

        self.log = print if verbose else (lambda *args: None)

        trace = self.ep_trace = defaultdict(list)
        trace.update({
                    'i_episode': self.i_episode,
                    'states': [],
                    'actions': [],
                    'rewards': [],
                    'finished': False,
                    'return': None
                })
        if reset:
            new_state = self.env.reset()
        else:
            new_state = self.env._state
        # trace['_state'] = self.env._state
        self._start_episode(new_state)
        done = False
        for i_step in range(max_steps):
            state = new_state

            self._render(render)
            action = self.policy.act(state)
            new_state, reward, done, info = self.env.step(action)
            self._experience(state, action, new_state, reward, done)
            
            trace['states'].append(state)
            trace['actions'].append(action)
            trace['rewards'].append(reward)

            if done:
                trace['finished'] = True
                self._render(render)
                break

        trace['states'].append(new_state)  # final state
        trace['return'] = sum(trace['rewards'])
        self._finish_episode(trace)
        self.i_episode += 1
        return dict(trace)

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

    def _start_episode(self, state):
        self.policy.start_episode(state)
        for vf in self.value_functions:
            vf.start_episode(state)

    def _finish_episode(self, trace):
        self.policy.finish_episode(trace)
        for vf in self.value_functions:
            vf.finish_episode(trace)
        

    def _experience(self, s0, a, s1, r, done):
        self.policy.experience(s0, a, s1, r, done)
        for vf in self.value_functions:
            vf.experience(s0, a, s1, r, done)

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


class Component(ABC):
    """A very abstract base class."""
    def __init__(self):
        super().__init__()
        self.agent = None
        self.saved = defaultdict(list)

    def experience(self, state, action, new_state, reward, done):
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

    def finish_episode(self, trace):
        """This function is run when an episode ends."""
        return

    def attach(self, agent):
        self.agent = agent

    @property
    def env(self):
        return self.agent.env

    @property
    def i_episode(self):
        return self.agent.i_episode

    @property
    def ep_trace(self):
        return self.agent.ep_trace

    def log(self, *args):
        self.agent.log(*args)

    def save(self, key, val):
        self.saved[key].append(val)


class Policy(Component):
    """Chooses actions"""
    def __init__(self):
        super().__init__()

    @abstractmethod
    def act(self, state):
        """Returns an action to take in the given state."""
        pass


class RandomPolicy(Policy):
    """Chooses actions randomly."""
    
    def act(self, state):
        return self.env.action_space.sample()

    def finish_episode(self, trace):
        self.ep_trace['berries'] = self.env._observe()[-1]


    class NeuralPolicyGradient(Policy):
        """Learns a policy directly."""
        def __init__(self):
            super().__init__()

        def attach(self, agent):
            super().attach(agent)
            model = self.model = Sequential()
            model.add(Dense(self.agent.n_actions, input_dim=len(self.env.reset()),
                            activation='softmax', kernel_initializer='glorot_uniform'))

            action = K.placeholder(shape=[None, self.action_size])
            discounted_rewards = K.placeholder(shape=[None, ])

            good_prob = K.sum(action * self.model.output, axis=1)
            eligibility = K.log(good_prob) * discounted_rewards
            loss = -K.sum(eligibility)

            optimizer = Adam(lr=self.learning_rate)
            updates = optimizer.get_updates(self.model.trainable_weights, [], loss)
            train = K.function([self.model.input, action, discounted_rewards], [], updates=updates)
            return train

            
    def softmax(x):
        ex = np.exp(x)
        return ex / ex.sum()



class ValueFunction(Component):
    """Learns values."""
    def __init__(self, learn_rate=.1, discount=1):
        self.discount = discount
        self.learn_rate = learn_rate

    def features(self, s):
        if hasattr(self.env, 'nS'):
            x = [0] * self.env.nS
            x[s] = 1
            return x
        else:
            return np.r_[1, s]

    def predict(self, s):
        return 0
     

class ActionValueFunction(ValueFunction):
    """Learns a linear Q function by SGD."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = None

    def attach(self, agent):
        super().attach(agent)
        sx = len(self.features(self.env.reset()))
        sy = self.agent.n_actions
        shape = (sx, sy)
        self.model = LinearSGD(shape, learn_rate=self.learn_rate)
    
    def experience(self, s0, a, s1, r, done):
        x0, x1 = self.features(s0), self.features(s1)
        target = self.model.predict(x1)
        target[a] = r + self.discount * np.max(self.model.predict(x1))
        self.model.update(x0, target)

    def predict(self, s):
        x = self.features(s)
        return self.model.predict(x)


class StateValueFunction(ValueFunction):
    """Learns a linear V function by SGD."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class LinearV(StateValueFunction):
    """Learns a linear V function by SGD."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = None

    def attach(self, agent):
        super().attach(agent)
        sx = len(self.features(self.env.reset()))
        sy = 1
        shape = (sx, sy)
        self.model = LinearSGD(shape, learn_rate=self.learn_rate)

    def experience(self, s0, a, s1, r, done):
        x = self.features(s0)
        # TODO: should we set target to 0 if done is True?
        target = r + self.discount * self.predict(s1)
        self.model.update(x, target)

    def predict(self, s):
        x = self.features(s)
        return self.model.predict(x)[0]
    
    def finish_episode(self, trace):
        self.ep_trace['theta_v'] = self.model.theta[:, 0].copy()


# from models import BayesianRegression
class BayesianRegressionV(StateValueFunction):
    """Learns a linear V function by SGD."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = None
        self.states = []
        self.returns = []

    def attach(self, agent):
        super().attach(agent)
        sx = len(self.features(self.env.reset()))
        self.model = BayesianRegression(np.zeros(sx), sigma_w=1)

    def predict(self, s):
        x = self.features(s)
        return self.model.predict(x)

    def finish_episode(self, trace):
        # import ipdb, time; ipdb.set_trace(); time.sleep(0.5)
        states = np.array([self.features(s).astype('float32') for s in trace['states']])
        returns = get_returns(trace['rewards'])
        returns.append(0) # return from final state

        self.states.extend(states)
        self.returns.extend(returns)

        self.model.update(states, returns)
        self.ep_trace['theta_v'] = self.model.w.copy()
        self.ep_trace['sigma_w'] = self.model.sigma_w.copy()

 


class TDLambdaV(StateValueFunction):
    """Learns a linear value function with TD lambda."""
    def __init__(self, trace_decay=0, **kwargs):
        super().__init__(**kwargs)
        self.trace_decay = trace_decay
        self.trace = None

    def attach(self, agent):
        self.agent = agent
        shape = len(self.features(self.env.reset()))
        self.trace = np.zeros(shape)
        self.theta = np.zeros(shape)
        self.theta_update = np.zeros(shape)

    def start_episode(self, state):
        self.theta = self.theta_update.copy()

    def experience(self, s0, a, s1, r, done):
        target = r + self.discount * self.predict(s1)
        x = self.features(s0)
        prediction = x @ self.theta
        error = target - prediction
        self.trace = self.trace_decay * self.trace + x
        self.theta_update += self.learn_rate * error * self.trace
        # self.theta_update *= self.decay

    def predict(self, s):
        x = self.features(s)
        return x @ self.theta

    def finish_episode(self, trace):
        self.ep_trace['theta_v'] = self.theta.copy()


class FixedV(StateValueFunction):
    """User-specified value function."""
    def __init__(self, theta):
        super().__init__()
        self.theta = np.array(theta)

    def predict(self, s):
        x = self.features(s)
        return x @ self.theta

    def finish_episode(self, trace):
        self.ep_trace['theta_v'] = self.theta.copy()


class MaxQPolicy(Policy):
    """Chooses the action with highest Q value."""
    def __init__(self, Q, epsilon=0.5, anneal=.95, **kwargs):
        super().__init__(**kwargs)
        self.Q = Q
        self.epsilon = epsilon
        self.anneal = anneal

    def act(self, state, anneal_step=0):
        epsilon = self.epsilon * self.anneal ** anneal_step
        if np.random.rand() < epsilon:
            return np.random.randint(self.env.n_actions)
        else:
            q = self.Q.predict(state)
            noise = np.random.random(q.shape) * .001
            return np.argmax(q + noise)


def interactions(x):
    return [a * b for a, b in it.combinations(x, 2)]

class MemV(StateValueFunction):
    """docstring for MemV"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tbl = defaultdict(list)

    def finish_episode(self, trace):
        returns = concatv(reversed(np.cumsum(list(reversed(trace['rewards'])))), [0])
        for s, r in zip(trace['states'], returns):
            self.tbl[s].append(r)


# from sklearn.linear_model import SGDRegressor
# class MonteCarloV(ValueFunction):
#     """Learns a linear value function with every-step Monte Carlo."""
#     def __init__(self, env, **kwargs):
#         super().__init__(env, **kwargs)
#         self.model = SGDRegressor()

#     def finish_episode(self, trace):
#         X = np.array([self.features(s) for s in trace['states']])
#         y = list(reversed(np.cumsum(list(reversed(trace['rewards'])))))
#         y.append(0)  # value of final state
#         self.model.partial_fit(X, y)

#     def predict(self, s):
#         if self.model.coef_ is not None:
#             return self.model.predict(self.features(s).reshape(1, -1))[0]
#         else:
#             return 0

#     @property
#     def theta(self):
#         return self.model.coef_



class Astar(Policy):
    """A* search finds the shortest path to a goal."""
    def __init__(self, heuristic):
        super().__init__()
        self.heuristic = heuristic
        self.plan = iter(())

    def start_episode(self, state):
        self.history = Counter()
        self.model = Model(self.env)

    def act(self, state):
        # return self.env.action_space.sample()
        self.history[state] += 1
        try:
            return next(self.plan)
        except StopIteration:
            self.plan = iter(self.make_plan(state))
            return next(self.plan)

    def eval_node(self, node):
        if not node.path:
            return np.inf  # the empty plan has infinite cost
        obs = self.env._observe(node.state)
        value = 0 if node.done else self.heuristic(self.env, obs) * 1.001
        boredom = - 0.1 * self.history[obs]
        score = node.reward + value + boredom
        return - score
    
    def make_plan(self, state, expansions=5000):
        
        Node = namedtuple('Node', ('state', 'path', 'reward', 'done'))
        eval_node = self.eval_node
        start = Node(self.env._state, [], 0, False)
        frontier = PriorityQueue(key=eval_node)
        frontier.push(start)
        reward_to_state = defaultdict(lambda: -np.inf)
        # import IPython; IPython.embed()
        best_finished = start

        def expand(node):
            # print(node.state, node.reward, self.rts[node.state], V(env._observe(node.state)))
            # time.sleep(0.1)
            nonlocal best_finished
            # best_finished = min((best_finished, node), key=eval_node)
            s0, p0, r0, _ = node
            for a, s1, r, done in self.model.options(s0):
                node1 = Node(s1, p0 + [a], r0 + r, done)
                if node1.reward <= reward_to_state[s1]:
                    # print('abandon')
                    pass
                    continue  # cannot be better than an existing node
                # self.save('node', node)
                reward_to_state[s1] = node1.reward
                if done:
                    best_finished = min((best_finished, node1), key=eval_node)
                else:
                    frontier.push(node1)
                    
        for i in range(expansions):
            self.save('frontier', [n[1].state for n in frontier])
            if frontier:
                expand(frontier.pop())
            else:
                break

        if frontier:
            # plan = min(best_finished, frontier.pop(), key=eval_node)
            plan = frontier.pop()
            raise RuntimeError('No plan found.')
        else:
            plan = best_finished
            print('best_finished = {}'.format(best_finished))
        # choices = concat([completed, map(get(1), take(100, frontier))])
        # plan = min(choices, key=eval_node(noisy=True))
        # self.log(
        #     i,
        #     len(plan.path), 
        #     -round(eval_node(plan, noisy=False), 2),
        #     plan.done,
        # )
        # self._trace['paths'].append(plan.path)
        self.save('plan', plan)
        return plan.path




class ValSearchPolicy(Policy):
    """Searches for the maximum reward path using a model."""
    def __init__(self, V, replan=False, epsilon=0, noise=1, anneal=1, **kwargs):
        super().__init__(**kwargs)
        self.V = V
        self.replan = replan
        self.epsilon = epsilon
        self.noise = noise
        self.anneal = anneal
        self.history = None
        self.model = None
        self.plan = None

    def start_episode(self, state):
        self.history = Counter()
        self.model = Model(self.env)
        self.plan = iter(())  # start with no plan

    def finish_episode(self, trace):
        self.ep_trace['berries'] = self.env._observe()[-1]

    def act(self, state):
        # return self.env.action_space.sample()
        self.history[state] += 1
        try:
            if self.replan:
                raise StopIteration()
            else:
                return next(self.plan)
        except StopIteration:
            self.plan = iter(self.make_plan(state))
            return next(self.plan)

    def make_plan(self, state, expansions=2000):

        Node = namedtuple('Node', ('state', 'path', 'reward', 'done'))
        env = self.env
        V = memoize(self.V.predict)
        self.node_history = []

        def eval_node(node, noisy=False):
            if not node.path:
                return np.inf  # the empty plan has infinite cost
            obs = env._observe(node.state)
            noise = np.random.rand() * (self.noise * self.anneal ** self.i_episode) if noisy else 0
            value = 0 if node.done else V(obs)[0]
            boredom = - 0.1 * self.history[obs]
            score = node.reward + value + noise + boredom
            return - score

        start = Node(env._state, [], 0, False)
        frontier = PriorityQueue(key=eval_node)
        frontier.push(start)
        reward_to_state = defaultdict(lambda: -np.inf)
        reward_to_state[start.state] = 0
        best_finished = start

        def expand(node):
            nonlocal best_finished
            best_finished = min((best_finished, node), key=eval_node)
            s0, p0, r0, _ = node
            for a, s1, r, done in self.model.options(s0):
                node1 = Node(s1, p0 + [a], r0 + r, done)
                if node1.reward <= reward_to_state[s1]:
                    continue  # cannot be better than an existing node
                self.node_history.append(
                    {'path': node1.path,
                     'r': node1.reward,
                     'b': self.env._observe(node1.state)[-1]    ,
                     'v': -eval_node(node1)})
                reward_to_state[s1] = node1.reward
                if done:
                    best_finished = min((best_finished, node1), key=eval_node)
                else:
                    frontier.push(node1)
                    
        for i in range(expansions):
            if frontier:
                expand(frontier.pop())
            else:
                break



        if frontier:
            plan = min(best_finished, frontier.pop(), key=eval_node)
        else:
            plan = best_finished
        # choices = concat([completed, map(get(1), take(100, frontier))])
        # plan = min(choices, key=eval_node(noisy=True))
        self.log(
            i,
            len(plan.path), 
            -round(eval_node(plan, noisy=False), 2),
            plan.done,
        )
        # self._trace['paths'].append(plan.path)
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




