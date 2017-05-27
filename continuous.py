import gym
from gym import spaces
import numpy as np
import pandas as pd
import itertools as it

from abc import ABC, abstractmethod
from utils import *
from copy import deepcopy

MAX_STEPS = 100
np.set_printoptions(precision=3)
API_KEY = 'sk_R6mkDKdZTMC3deGMZi1Slg'





class EasyEnv(gym.Env):
    def __init__(self, wind=0, speed=.1):
        self.wind = wind
        self.speed = speed

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-1, 1, shape=(1,))

    def _step(self, action):
        self.state = self.state + (action-0.5) * self.speed + self.wind
        done = not (-1 < self.state < 1)
        reward = 100 if done else -1
        return self.state, reward, done, {}

    def _reset(self):
        self.state = 0
        return self.state


class Agent(ABC):
    """An agent that can run on  discrete openai gym environments.

    All agents inherit from this abstract base class (which itself cannot be
    instantiated). A class implementing Agent must override act. Any learning
    algorithm (e.g. Sarsa) will also implement update."""
    def __init__(self, env, discount=0.99):
        self.env = env
        self.i_episode = 0
        self.discount = discount

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

    def run_episode(self, render=False, max_steps=1000, interact=False, verbose=False):
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

    def run_many(self, n_episodes, **kwargs):
        """Runs several episodes, returns a summary of results."""
        data = {
            'i_episode': [],
            'n_steps': [],
            'return': [],
            'finished': [],
        }
        for _ in range(n_episodes):
            trace = self.run_episode(**kwargs)
            data['i_episode'].append(trace['i_episode'])
            data['n_steps'].append(len(trace['states']))
            data['return'].append(trace['return'])
            data['finished'].append(trace['finished'])

        return data

    def _render(self, mode):
        if mode == 'step':
            input('> ')
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



# ========================== #
# ========= Agents ========= #
# ========================== #

class RandomAgent(Agent):
    """A not-too-bright Agent."""
    def __init__(self, env):
        super().__init__(env)
    
    def act(self, state):
        return self.env.action_space.sample()

class LinearModelAgent(Agent):
    def __init__(self, env, weights=None, kernel=None):
        super().__init__(env)
        self.weights = weights or np.random.randn(env.observation_space.shape[0] + 1)
        self.kernel = kernel

    def act(self, observation):
        if callable(self.kernel):
            observation = self.kernel(observation)
        observation = np.r_[observation, 1]
        return int(np.dot(self.weights, observation) > 0)

    def __str__(self):
        return ('LinearModelAgents(weights={weights}, kernel={kernel})'
                .format_map(self.__dict__))




class LinearModel(object):
    """Learns a linear model by SGD."""
    def __init__(self, shape, learn_rate, decay=0):
        self.shape = self.phi(np.random.random(shape)).shape
        self.learn_rate = learn_rate
        self.theta = np.random.random(self.shape)
        self.trace = np.zeros(self.shape)

    def update(self, x, y):
        yhat = x @ self.theta
        error = y - yhat
        self.trace = self.decay * self.trace + x
        self.theta += self.learn_rate * error * self.trace

    def phi(self, x):
        return np.r_[1, x]

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
        shape = (env.observation_space.shape[0], env.action_space.n)
        self.Q = LinearModel(shape, learn_rate)


    def act(self, state):
        if self.exploration:
            epsilon = self.epsilon * self.anneal ** self.i_episode
            if self.exploration == 'epsilon':
                if np.random.rand() < epsilon:
                    return np.random.randint(self.Q.shape[1])
                else:
                    noise = np.random.randn(self.Q.shape[1]) * .001
                    return np.argmax(self.Q(state) + noise)

            elif self.exploration == 'noise':
                noise = np.random.randn(self.Q.shape[1]) * epsilon
                return np.argmax(self.Q(state) + noise)

    def update(self, s0, a, s1, r, done):
        # self.Q.theta = np.array([[-1, 1]])
        # return
        target = self.Q(s0)
        target[a] = r + self.discount * np.max(self.Q(s1))
        self.Q.update(s0, target)


class ValueSearch(Agent):
    """Learns expected values of taking an action in a state."""
    def __init__(self, env, learn_rate=.85, discount=.99, epsilon=.5, anneal=.99,
                 exploration='epsilon'):
        super().__init__(env, discount=discount)
        self.epsilon = epsilon
        self.anneal = anneal
        self.exploration = exploration
        shape = env.observation_space.shape[0]
        self.V = LinearModel(shape, learn_rate)


    def act(self, state):
        return self.env.action_space.sample()

    def search(self, state):

        def expand(env, depth, path, rerward):
            env = deepcopy(self.env)
            for a in env.action_space:
                




    def update(self, s0, a, s1, r, done):
        target = r + self.discount * self.V(s1)
        self.V.update(s0, target)


class TDLambda(Agent):
    """Learns expected values of taking an action in a state."""
    def __init__(self, env, learn_rate=.85, discount=.99, epsilon=.5, anneal=.99,
                 exploration='epsilon', lam=0.5):
        super().__init__(env, discount=discount)
        self.epsilon = epsilon
        self.anneal = anneal
        self.exploration = exploration
        self.lam = lam
        shape = self.phi(env.observation_space.sample()).shape
        self.e = np.zeros(shape)
        self.V = LinearModel(shape, learn_rate)

    def phi(self, x):
        return np.r_[x, abs(x)]

    def act(self, state):
        return self.env.action_space.sample()

    def update(self, s0, a, s1, r, done):
        s0, s1 = self.phi(s0), self.phi(s1)
        target = r + self.discount * self.V(s1)
        self.e = self.discount * self.lam * self.e + s0
        error = target - self.V(s0)
        self.V.theta += self.V.learn_rate * error * self.e

# class TDLambda(Agent):
#     """Learns expected values of taking an action in a state."""
#     def __init__(self, env, learn_rate=.85, discount=.99, epsilon=.5, anneal=.99,
#                  exploration='epsilon', lam=0.5):
#         super().__init__(env, discount=discount)
#         self.epsilon = epsilon
#         self.anneal = anneal
#         self.exploration = exploration
#         self.lam = lam
#         shape = self.phi(env.observation_space.sample()).shape
#         self.e = np.zeros(shape)
#         self.V = LinearModel(shape, learn_rate)

#     def phi(self, x):
#         return np.r_[x, abs(x)]

#     def act(self, state):
#         return self.env.action_space.sample()

#     def update(self, s0, a, s1, r, done):
#         s0, s1 = self.phi(s0), self.phi(s1)
#         target = r + self.discount * self.V(s1)
#         self.e = self.discount * self.lam * self.e + s0
#         error = target - self.V(s0)
#         self.V.theta += self.V.learn_rate * error * self.e




def quadratic_agent(env):

    def kernel(x):
        return np.concatenate((x, x**2))

    size = env.observation_space.shape[0]
    weights = np.random.randn(size * 2)
    
    return LinearModelAgent(weights, kernel)

def interact_agent(env):

    def kernel(x):
        interactions = list(map(np.product, it.combinations(x, r=2)))
        return np.concatenate((x, interactions))
    
    size = env.observation_space.shape[0]
    obs = kernel(np.zeros(size))
    weights = np.random.randn(len(obs))
    
    return LinearModelAgent(weights, kernel)




# ======================================= #
# ========= Learning algorithms ========= #
# ======================================= #


def random_guessing(env, create_agent, iterations=100) -> Agent:
    
    best_agent = None
    best_score = 0
    scores = []

    for i_episode in range(iterations):
        agent = create_agent(env)
        score, finished = run_episode(agent, env, max_steps=MAX_STEPS)
        
        if score > best_score:
            best_score = score
            best_agent = agent
            scores.append(best_score)


    return best_agent, best_score, scores


def hill_climbing(agent, iterations=100, rate=0.1):
    best_score = 0
    scores = []

    for i_episode in range(iterations):
        # assume agent has a weights parameter
        delta = rate * np.random.randn(agent.weights.shape[0])
        agent.weights += delta

        score = agent.run_episode(max_steps=MAX_STEPS)['return']
        scores.append(score)

        if score >= best_score:
            best_score = score
        else:
            agent.weights -= delta

    return scores





if __name__ == '__main__':
    # env = gym.make('CartPole-v0')
    env = EasyEnv()

    # pd.Series(random_guessing(env, quadratic, 1000)[2])
    agent, _, scores = hill_climbing(env, linear_agent, 10)
    print(scores)
    

    #df = pd.DataFrame({
    #    'linear': [random_guessing(env, linear_agent)[1]
    #               for _ in range(5)],
    #    'interaction': [random_guessing(env, interact_agent)[1]
    #                    for _ in range(5)],
    #})
    