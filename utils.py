from IPython.display import clear_output

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
class Heap(list):
        
    def pop(self) :
        return heapq.heappop(self)
        
    def push(self, item) :
        heapq.heappush(self, item)
        
    def pushpop(self, item) :
        return heapq.heappushpop(self, item)
        
    def poppush(self, item) :
        return heapq.replace(self, item)