import gym
import numpy as np
import pandas as pd
import itertools as it

from utils import *

MAX_STEPS = 200
np.set_printoptions(precision=3)
API_KEY = 'sk_R6mkDKdZTMC3deGMZi1Slg'

# ========================== #
# ========= Agents ========= #
# ========================== #

class Agent(object):
    """An intelligent agent"""
    def act(self, observation):
        raise NotImplementedError()


class LinearModelAgent(Agent):
    def __init__(self, weights, kernel=None):
        self.weights = weights
        self.kernel = kernel

    def act(self, observation):
        if callable(self.kernel):
            observation = self.kernel(observation)
        return int(np.dot(self.weights, observation) > 0)

    def __str__(self):
        return ('LinearModelAgents(weights={weights}, kernel={kernel})'
                .format_map(self.__dict__))


def linear_agent(env):
    weights = np.random.randn(env.observation_space.shape[0])
    return LinearModelAgent(weights)


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


def hill_climbing(env, create_agent, iterations=100, rate=0.1) -> Agent:
    agent = create_agent(env)
    best_score = 0
    scores = []

    for i_episode in range(iterations):
        # assume agent has a weights parameter
        delta = rate * np.random.randn(agent.weights.shape[0])
        agent.weights += delta

        score, finished = run_episode(agent, env, max_steps=MAX_STEPS)

        if score > best_score:
            best_score = score
            scores.append(best_score)
        else:
            agent.weights -= delta

    return agent, best_score, scores





if __name__ == '__main__':
    env = gym.make('CartPole-v0')

    rand = pd.Series(random_guessing(env, linear_agent, 1000)[2])
    climb = pd.Series(hill_climbing(env, linear_agent, 1000)[2])

    #df = pd.DataFrame({
    #    'linear': [random_guessing(env, linear_agent)[1]
    #               for _ in range(5)],
    #    'interaction': [random_guessing(env, interact_agent)[1]
    #                    for _ in range(5)],
    #})
    