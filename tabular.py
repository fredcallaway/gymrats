"""Agents that operate in discrete fully observable environments."""

from collections import namedtuple, defaultdict, Counter, deque
import numpy as np
from abc import ABC, abstractmethod
import utils
import time
np.set_printoptions(precision=3, linewidth=200)


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
        """This function is run once when an episode begins, starting at state.

        This can be used to e.g. to initialize episode-specific memory as necessary
        for n-step TD learning."""
        pass

    def run_episode(self, render=False, max_steps=1000, interact=False,
                    verbose=False):
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


class RandomAgent(Agent):
    """A not-too-bright Agent."""
    def __init__(self, env):
        super().__init__(env)
    
    def act(self, state):
        return self.env.action_space.sample()
      

class HumanAgent(Agent):
    """Keyboard controlled agent."""
    def __init__(self, env, actions=None):
        super().__init__(env)
        if not actions:
            actions = list(map(str, range(env.nA)))
        self.actions = dict(zip(actions, range(len(actions))))

    def run_episode(self, **kwargs):
        # kwargs['render'] = 'clear'
        # kwargs['render'] = 'human'
        return super().run_episode(**kwargs)

    def update(self, s0, a, s1, r, done):
        msg = utils.join(s0, a, s1, r, done)
        print(msg)


    def act(self, state):
        char = None
        while True:
            print('> ', end='', flush=True)
            char = self.read_char()
            if char == 'd':
                print('Exiting.')
                exit()
            if char in self.actions:
                return self.actions[char]
            else:
                print('Actions: ', ' '.join(map(str, self.actions)), '(d to exit)')

    @staticmethod
    def read_char():
        import tty, sys, termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch


class _QLearningAgent(Agent):
    """Learns expected values of taking an action in a state."""
    def __init__(self, env, learn_rate=.85, discount=.99, epsilon=.5, anneal=.99,
                 exploration='epsilon'):
        super().__init__(env, discount=discount)
        self.learn_rate = learn_rate
        self.epsilon = epsilon
        self.anneal = anneal
        self.exploration = exploration

        # Q is a table of estimated values for taking an action in a state. It
        # incorporates the direct reward as well as expected future reward.
        self.Q = np.zeros((self.n_states, self.n_actions))

    @property
    def V(self):
        return np.max(self.Q, axis=1)

    def act(self, state):
        if self.exploration:
            epsilon = self.epsilon * self.anneal ** self.i_episode
            if self.exploration == 'epsilon':
                if np.random.rand() < epsilon:
                    return np.random.randint(self.n_actions)
                else:
                    noise = np.random.randn(self.Q.shape[1]) * .001
                    return np.argmax(self.Q[state] + noise)

            elif self.exploration == 'noise':
                noise = np.random.randn(self.Q.shape[1]) * epsilon
                return np.argmax(self.Q[state] + noise)

    def update(self, s0, a, s1, r, done):
        # Update Q table.
        learned_value = r + self.discount * np.max(self.Q[s1])
        self.Q[s0, a] += self.learn_rate * (learned_value - self.Q[s0, a])



class Policy(ABC):
    """Chooses actions."""
    def __init__(self, env, epsilon=0, anneal=1):
        self.env = env
        self.epsilon = epsilon
        self.anneal = anneal

    @abstractmethod
    def __call__(self, state):
        """Returns an action to take in a given state."""
        pass


class MaxQPolicy(Policy):
    """Chooses the action with highest Q value."""
    def __init__(self, env, Q, **kwargs):
        super().__init__(env, **kwargs)
        self.Q = Q

    def __call__(self, state, anneal_step=0):
        epsilon = self.epsilon * self.anneal ** anneal_step
        if np.random.rand() < epsilon:
            return np.random.randint(self.env.n_actions)
        else:
            noise = np.random.randn(self.Q.shape[1]) * .001
            return np.argmax(self.Q[state] + noise)


class QLearningAgent(Agent):
    """Learns expected values of taking an action in a state."""
    def __init__(self, env, learn_rate=1, discount=1, 
                 policy=MaxQPolicy, policy_kws=dict(epsilon=0.5, anneal=.99)):
        super().__init__(env, discount=discount)
        self.learn_rate = learn_rate

        # Q is a table of estimated values for taking an action in a state. It
        # incorporates the direct reward as well as expected future reward.
        self.Q = np.zeros((self.n_states, self.n_actions))
        self.policy = policy(env, self.Q, **policy_kws)

    @property
    def V(self):
        return np.max(self.Q, axis=1)

    def act(self, state):
        return self.policy(state, self.i_episode)

    def update(self, s0, a, s1, r, done):
        # Update Q table.
        learned_value = r + self.discount * np.max(self.Q[s1])
        self.Q[s0, a] += self.learn_rate * (learned_value - self.Q[s0, a])


class NstepSarsa(QLearningAgent):
    """7.2 in Sutton and Barto"""
    def __init__(self, env, nstep=1, **kwargs):
        super().__init__(env, **kwargs)
        self.nstep = nstep

    def start_episode(self, state):
        self.memory = deque(maxlen=self.nstep)

    def update(self, s0, a, s1, r, done):
        self.log('update', s0, a)
        self.log('  memory', list(self.memory))

        def update(s_t, a_t, finished=False):
            # value is G in Sutton p. 157
            rewards = sum(m[2] * self.discount ** i for i, m in enumerate(self.memory))
            future_value = rewards + (self.discount ** self.nstep) * self.Q[s0, a]
            if finished:
                value = rewards
                self.log('Q{} = {}'.format((s_t, a_t), rewards))
            else:
                value = rewards + future_value
                self.log('Q{} = {} + Q{} = {}'.format((s_t, a_t), rewards, (s0, a), value))

            self.Q[s_t, a_t] += self.learn_rate * (value - self.Q[s_t, a_t])

        if len(self.memory) == self.nstep:
            s_t, a_t, _ = self.memory[0]
            update(s_t, a_t)
        
        self.memory.append((s0, a, r))

        if done:
            self.log('finish remaining')
            while self.memory:
                s_t, a_t, _ = self.memory[0]
                update(s_t, a_t, finished=True)
                self.memory.popleft()






        


class SarsaAgent(QLearningAgent):
    def start_episode(self, state):
        # Store last state, action, reward because updates for step t are done
        # after choosing action of t+1
        self.last_sar = None

    def update(self, state, action, new_state, reward, done):

        # We update the action at the last time step based on 
        # the outcome of this time step.
        s1, a1 = state, action

        if self.last_sar is not None:
            s0, a0, r = self.last_sar
            learned_value = r + self.discount * self.Q[s1, a1]
            self.Q[s0, a0] += self.learn_rate * (learned_value - self.Q[s0, a0])

        if done:
            # There won't be another action, so we update immediately. Value of final
            # state is necessarily 0.
            learned_value = reward + 0
            self.Q[s1, a1] += self.learn_rate * (learned_value - self.Q[s1, a1])

        self.last_sar = (state, action, reward)




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

    def __init__(self, env, depth=None, model=None, 
                 V=None, memory=False, replan=False, **kwargs):
        super().__init__(env, replan=replan, **kwargs)
        if depth is None:
            depth = -1  # infinite depth
        self.depth = depth

        if model is None:
            model = TrueModel(env)
        self.model = model
        self.memory = memory
        self.last_state = None
        self.explored = set()

    def reward(self, s0, a, s1, r):
        return r

    def act(self, s0):
        self.explored.add(s0)
        return super().act(s0)

    def update(self, s0, a, s1, r, done):
        if self.memory:
            # TODO longer memory (should take into account recency of
            # a repeated state).
            self.last_state = s0

    def make_plan(self, state):
        def eval_path(path):
            # Choose path with greatest reward. In case of a tie, prefer the path
            # that takes you to unexplored territory. If no such path exists,
            # don't go back to the state you were at previously.
            reward = sum((self.reward(*node[1:])) * self.discount ** i
                         for i, node in enumerate(path))
            num_new = sum(node.s1 not in self.explored for node in path)
            not_backwards = all(node.s1 != self.last_state for node in path)
            return (reward, num_new, not_backwards)
        path = max(self.model.paths(state, depth=self.depth), key=eval_path)
        return (node.a for node in path)


class PseudoAgent(PlanAgent):
    """Searches for the maximum reward path using a model."""

    def __init__(self, env, depth=None, model=None, 
                 pseudo_freq=0, pseudo_mode='full', pseudo_weight=1, pseudo_rewards=None,
                 V=None, memory=False, replan=False, **kwargs):
        super().__init__(env, replan=replan, **kwargs)
        if depth is None:
            depth = -1  # infinite depth
        self.depth = depth

        if model is None:
            model = TrueModel(env)
        self.model = model

        if pseudo_rewards:
            self.pseudo_rewarder = PrecomputedPseudoRewarder(pseudo_rewards)
        elif pseudo_freq:
            if V is None:
                V = value_iteration(env, self.discount)
            self.pseudo_rewarder = PseudoRewarder(model, V, pseudo_freq, pseudo_mode,
                                                  pseudo_weight, self.discount, )
        else:
            self.pseudo_rewarder = None

        self.memory = memory
        self.last_state = None

    def start_episode(self, state):
        if self.pseudo_rewarder:
            self.pseudo_rewarder.start_episode(state)
        super().start_episode(state)

    def reward(self, s0, a, s1, r):
        if self.pseudo_rewarder:
            pseudo = self.pseudo_rewarder.recover(s0, a, s1)
        else:
            pseudo = 0
        return r + pseudo

    def update(self, s0, a, s1, r, done):
        if not done and self.pseudo_rewarder:
            self.pseudo_rewarder.update(s1)
        if self.memory:
            # TODO longer memory (should take into account recency of
            # a repeated state).
            self.last_state = s0

    def make_plan(self, state):
        def eval_path(path):
            # Choose path with greatest reward. In case of a tie, prefer the path
            # that takes you to unexplored territory. If no such path exists,
            # don't go back to the state you were at previously.
            reward = sum((self.reward(*node[1:])) * self.discount ** i
                         for i, node in enumerate(path))
            num_new = sum(node.s1 not in self.explored for node in path)
            not_backwards = all(node.s1 != self.last_state for node in path)
            return (reward, num_new, not_backwards)

        path = max(self.model.paths(state, depth=self.depth), key=eval_path)
        return (node.a for node in path)


class PseudoRewarder(object):
    """Doles out pseudo-rewards."""
    def __init__(self, model, V, freq, mode, weight, discount):
        assert mode in ('horizon', 'full')

        self.model = model
        self.V = V
        self.freq = freq
        self.mode = mode
        self.weight = weight
        self.discount = discount

    def recover(self, s0, a, s1):
        return self.weight * self._cache.get(s1, 0)

    def start_episode(self, s0):
        self._cache = {}
        if self.mode == 'horizon':
            self.compute_horizon(s0)
        elif self.mode == 'full':
            self.compute_full(s0)

    def update(self, s1):
        if self.mode == 'horizon' and s1 in self._cache:
            # Whenever we get a pseudo-reward, we compute new ones.
            self._cache = {}
            self.compute_horizon(s1)

    def horizon(self, s0):
        for path in self.model.paths(s0, depth=self.freq):
            yield path[-1].s1

    def compute_horizon(self, s0):
        # TODO: handle different length paths
        # TODO: handle multiple paths to one state
        # import ipdb, time; ipdb.set_trace(); time.sleep(0.5)
        for s1 in self.horizon(s0):
            assert s1 not in self._cache
            self._cache[s1] = self.discount * self.V[s1] - self.V[s0]

    def compute_full(self, s0):
        for s1 in self.horizon(s0):
            self._cache[s1] = self.discount * self.V[s1] - self.V[s0]
            self.compute_full(s1)  # TODO use queue not recursion


class PrecomputedPseudoRewarder(object):
    """Doles out precomputed pseudo-rewards."""
    def __init__(self, rewards):
        self.rewards = rewards
    
    def recover(self, s0, a, s1):
        return self.rewards[s0][s1]

    def start_episode(self, s0):
        return

    def update(self, s1):
        return




class Model(object):
    """Learned model of an MDP."""
    Node = namedtuple('Node', ['p','s0', 'a', 's1', 'r'])

    def __init__(self, env):
        # (s, a) -> [total_count, outcome_counts]
        self.env = env
        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n
        self.counts = defaultdict(lambda: [0, Counter()])

    def results(self, s, a):
        """Yields possible outcomes of taking action a in state s.

        Outcome is (prob, s1, r, done).
        """
        total_count, outcomes = self.counts[s, a]
        for (s1, r, done), count in outcomes.items():
            yield (count / total_count, s1, r, done)

    def expected_reward(self, s0, a, s1=None):
        return sum(p * r for (p, s1_, r, done) in self.results(s0, a)
                   if s1 is None or s1 == s1_)

    def update(self, s0, a, s1, r, done):
        self.counts[s0, a][0] += 1  # total count
        self.counts[s0, a][1][s1, r, done] += 1  # outcome count

    def paths(self, state, depth=-1, cycles=False):
        """Yields all paths to a final state or a state `depth` steps away."""

        def expand(path, explored, depth):
            # TODO explored!
            s0 = path[-1].s1 if path else state  # initial call
            for a in range(self.n_actions):
                for (p, s1, r, done) in self.results(s0, a):
                    if p and (cycles or s1 not in explored):
                        new_path = path + [self.Node(p, s0, a, s1, r)]
                        if done or depth == 1:
                            yield new_path
                        else: 
                            yield from expand(new_path, explored | {s1}, depth-1)

        yield from expand([], {state}, depth)


class TrueModel(Model):
    """Accurate model of a DiscreteEnv."""
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)

    def results(self, state, action):
        yield from self.env.P[state][action]

    def update(*args):
        pass


class ModelBasedAgent(Agent):
    """Agent that builds a model of the MDP."""
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        self.model = Model(env)
        self.Q = np.zeros((self.model.n_states, self.model.n_actions))

    def act(self, state):
        noise = np.random.randn(self.Q.shape[1]) / (self.i_episode + 1)
        return np.argmax(self.Q[state] + noise)  # a = action

    def update(self, s0, a, s1, r, done):
        self.model.update(s0, a, s1, r, done)
        self.update_policy(s0, a)

    def update_policy(self, s, a):
        Q, V = self.Q, self.V
        expected_future_reward = sum(p * V(s1) for p, s1, r, done in self.model.results(s, a))
        Q[s, a] = self.model.expected_reward(s, a) + self.discount * expected_future_reward

    def V(self, s):
        """The value of taking the best possible action from state s."""
        return self.Q[s].max()


class PrioritizedSweeping(ModelBasedAgent):
    """Learns by replaying past experience.

    https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume4/kaelbling96a-html/node29.html
    """
    def __init__(self, env, n_simulate=0, **kwargs):
        super().__init__(env, **kwargs)
        self.n_simulate = n_simulate
        self.value_change = np.zeros(self.n_states)
        self.predecessors = defaultdict(set)

    def update(self, s0, a, s1, r, done):
        super().update(s0, a, s1, r, done)

        self.predecessors[s1].add(s0)
        # Update Q by simulation.
        for _ in range(self.n_simulate):
            s = self.value_change.argmax()
            if self.value_change[s] == 0:
                break  # no new information to propogate
            self.value_change[s] = 0  # reset
            self.update_policy(s, self.act(s))

    def update_policy(self, s0, a):
        # Track changes to prioritize simulations.
        old_val = self.V(s0)
        super().update_policy(s0, a)
        change = abs(self.V(s0) - old_val)
        if change:
            self.log('change', s0, a, change)
        for s_pred in self.predecessors[s0]:
            self.value_change[s_pred] = max(self.value_change[s_pred], change)  # TODO weight by transition prob


def value_iteration(env, discount=.999, epsilon=0.001, max_iters=100000):
    """Returns the optimal value table for env."""
    V1 = np.zeros(env.observation_space.n)

    def value(result):
        # [(0.5, 0, 0, False), (0.5, 0, 0, False)] -> float
        return sum(p * (r + discount * V[s1])
                   for p, s1, r, _ in result)

    for i in range(1, max_iters+1):
        V = V1.copy()
        delta = 0
        for state, actions in env.P.items():

            # Example actions object. Keys are actions, values are
            # lists of (prob, next_state, reward, done).
            # {0: [(1.0, 1, 0, False)],
            #  1: [(1.0, 2, 0, False)],
            #  2: [(1.0, 3, -1, False)],
            #  3: [(1.0, 0, 4, False)]}

            results = actions.values()
            V1[state] = max(map(value, results))
            delta = max(delta, abs(V1[state] - V[state]))

        if delta < epsilon * (1 - discount) / discount:
            return V

    print('NOT CONVERGED')
    return V



def v_to_q(env, V):
    
    def rval(result):
        p, s1, r, _ = result
        return p * (r + V[s1])
    
    def qval(s, a):
        return sum(rval(result) for result in env.P[s][0])
    
    Q = np.array([[qval(s, a) for a in range(env.n_actions)]
                  for s in range(env.n_states)])
    return Q







