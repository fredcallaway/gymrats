
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

