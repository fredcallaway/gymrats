from IPython.display import clear_output
import itertools as it
def join(*args, sep=' '):
    return sep.join(map(str, args))


def run_episode(agent, env, max_steps=500, render=False):
    observation = env.reset()
    score = 0

    for _ in range(max_steps):
        action = agent.act(observation)
        if render:
            env.render()
        observation, reward, done, info = env.step(action)
        score += reward
        if done:
            break
    
    return score, done


def logged(condition=lambda r: True):
    def decorator(func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            if condition(result):
                print(func.__name__, args, kwargs, '->', result)
            return result
        return wrapper
    return decorator



def clear_screen():
    print(chr(27) + "[2J")
    clear_output()


def show_path(env, trace, render='human'):
    env._state = trace['_state']
    env.render(mode=render)
    for a in trace['actions']:
        env.step(a)
        env.render(mode=render)


import heapq
class PriorityQueue(list):
    def __init__(self, key):
        self.key = key

    def pop(self):
        return heapq.heappop(self)[1]
        
    def push(self, item):
        heapq.heappush(self, (self.key(item), item))

def dict_product(d):
    """All possible combinations of values in lists in `d`"""
    for k, v in d.items():
        if not isinstance(v, list):
            d[k] = [v]

    for v in list(it.product(*d.values())):
        yield dict(zip(d.keys(), v))