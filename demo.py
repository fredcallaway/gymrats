from discrete import *
from envs import *

class Demo(object):
    """Functions for demonstration."""

    def __call__(self, demo, seed=None, **kwargs):
        if seed is None:
            seed = np.random.randint(1000)
        print('seed =', seed)
        np.random.seed(seed)

        func = getattr(self, demo)
        return func(**kwargs)
      
    def sweep(self, ):
        env = MazeEnv('maze_19_1.json')
        subj_id = it.count()

        def make_data():
            for n_sim in [0, 1, 5, 10, 100]:
                for _ in range(10):
                    subj = next(subj_id)
                    agent = PrioritizedSweeping(env, n_simulate=n_sim)
                    data = agent.run_many(30)
                    for i, n_step in enumerate(data['n_steps']):
                        yield {'i': i, 'n_sim': n_sim, 'subj': subj, 'n_steps': n_step}

        data = pd.DataFrame(make_data())
        print(data['return'][-10:])
        return locals()


    def value(self, ):
        # env = MazeEnv('maze_9_1.json')
        mdp = GridMDP(5, 5, 1)
        print(mdp.grid)
        env = GridEnv(mdp)

        V = value_iteration(env)
        V = V.reshape(env.grid.shape)

        agent = QLearningAgent(env)
        data = agent.run_many(100)
        A = agent.V.reshape(env.grid.shape)

        print(V)
        print()
        print(A)


    def search(self, ):
        env = MazeEnv('maze_19_1.json')
        agent = SearchAgent(env)
        trace = agent.run_episode(render='step')
        print('reward:', sum(trace['rewards']))

        return locals()


    def foo(self, ):
        env = GridEnv(5, 5)
        agent = SearchAgent(env)
        trace = agent.run_episode()
        print('reward:', sum(trace['rewards']))

        return locals()


    def human(self, ):
        # env = MazeEnv('maze_19_1.json')
        from gym.envs.toy_text.taxi import TaxiEnv 
        env = TaxiEnv()
        agent = HumanAgent(env, 'uien')
        agent.run_episode()


    def sarsa(self, ):
        env = GridEnv(5, 5)
        # env = DecisionTreeEnv()

        sarsa = SarsaAgent(env)
        sarsa_data = sarsa.run_many(1000)
        print(sarsa_data['return'][-10:])

        ql = QLearningAgent(env)
        ql_data = ql.run_many(1000)
        print(ql_data['return'][-10:])
        
        rand = RandomAgent(env)
        rand_data = rand.run_many(10)
        print(rand_data['return'][-10:])




        return locals()


    def model(self, ):
        mdp = GridMDP(5, 5, 1)
        env = GridEnv(mdp)
        
        agent = ModelBasedAgent(env)
        data = agent.run_many(100)

        return locals()


    def grid_pseudo(self, seed=None, size=10, discount=0.99):
        # not same last_pseudo for 234

        size = 10
        discount = 0.99
        env = GridEnv(GridMDP(size, size, seed=seed))

        V = value_iteration(env, discount, epsilon=0.001)
        V = V.reshape((size, size))
        for i in range(len(V)):
            for j in range(len(V)):
                if (i+j) % 2:
                    V[i, j] = 0
        V = V.ravel()

        agents = {}
        traces = {}

        for depth in (1, 2, 3, 4, 5, 6):
            agents[depth] = SearchAgent(env, depth=depth, V=V, pseudo=True)
            traces[depth] = agents[depth].run_episode()
            print(depth, sum(traces[depth]['rewards']))

        return locals()


    def pseudo(self, discount=0.99, make_env=None, n_env=1, n_episode=1, 
               freqs=range(7), depths=range(1,7)):
        if make_env is None:
            make_env = lambda: DecisionTreeEnv.random(depth=12)

        def make_data():
            for i_env in range(n_env):
                env = make_env()
                V = value_iteration(env, discount, epsilon=0.001)

                for pseudo_freq in freqs:    
                    for depth in depths:
                        agent = SearchAgent(env, depth=depth, V=V, discount=discount,
                                            pseudo_freq=pseudo_freq)
                        for i_episode in range(n_episode):
                            trace = agent.run_episode()
                            yield {'i_episode': i_episode,
                                   'i_env': i_env,
                                   'pseudo_freq': pseudo_freq,
                                   'depth': depth, 
                                   'reward': sum(trace['rewards'])}

        df = pd.DataFrame(make_data())
        return locals()


    def decision(self, ):
        env = DecisionTreeEnv.random(8)

        config = [
            (False, None),
            (False, 1),
            (False, 2),
            (False, 5),
            (True, 1),
            (True, 2),
            (True, 5),
        ]

        for pr, depth in config:
            print('pseudo =', pr, '  depth =', depth)
            agent = SearchAgent(env, depth=depth, pseudo=pr)
            trace = agent.run_episode()
            print(sum(trace['rewards']))

        return locals()


    def depth(self, ):
        seed = np.random.randint(0, 10000)
        print('seed =', seed)
        mdp = GridMDP(5, 5, seed=seed)
        env = GridEnv(mdp)
        # print(mdp.grid)

        config = [
            (None, False),
            (1, False),
            (1, True),
            (2, True),
            (5, True),
        ]

        for depth, pr in config:
            print('depth =', depth, ';  pseudo =', pr)
            agent = SearchAgent(env, depth=depth, pseudo=pr)
            trace = agent.run_episode()
            print(sum(trace['rewards']))
            # print(trace['states'])


    def plane(self, ):
        env = DeterministicGraphEnv('plane')
        agents = {
            'qlearn': QLearningAgent(env),
            'q2': QLearningAgent(env),
        }
            
        seed = np.random.randint(1000)
        print('seed =', seed)
        np.random.seed(seed)

        # traces = [agent.run_episode() for _ in range(20)]
        def data():
            for name, agent in agents.items():
                env.seed(0)
                np.random.seed(0)
                for _ in range(100):
                    trace = agent.run_episode()
                    yield {'agent': name, 'i_ep': trace['i_episode'], 
                           'reward': sum(trace['rewards']), 'len': len(trace['actions'])}
        
        df = pd.DataFrame(data())
        return locals()


if __name__ == '__main__':
    demo = Demo()
    func = eval('demo("{}")'.format(sys.argv[1]))

    # locs = func()
    # code = '\n'.join("{} = locs['{}']".format(k, k) for k in locs.keys())
    # exec(code)
    