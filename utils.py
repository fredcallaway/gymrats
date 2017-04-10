from IPython.display import clear_output

def join(*args, sep=' '):
    return sep.join(map(str, args))


def run_episode(agent, env, max_steps=500):
    observation = env.reset()
    score = 0

    for _ in range(max_steps):
        action = agent.act(observation)
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


def show_path(env, actions, state=None, render='step'):
    if state is None:
        state = env.reset()
    env.render(mode=render)
    for a in actions:
        env.step(a)
        env.render(mode=render)