from collections import namedtuple, defaultdict, Counter, deque
import itertools as it
import numpy as np
from abc import ABC, abstractmethod
from utils import clear_screen, PriorityQueue
import time
np.set_printoptions(precision=3, linewidth=200)

from tqdm import tqdm, trange, tnrange
from copy import deepcopy

from agents import Component


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


# class MemV(StateValueFunction):
#     """docstring for MemV"""
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.tbl = defaultdict(list)

#     def finish_episode(self, trace):
#         returns = concatv(reversed(np.cumsum(list(reversed(trace['rewards'])))), [0])
#         for s, r in zip(trace['states'], returns):
#             self.tbl[s].append(r)



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



