"""Exact solutions for tabular MDPs."""

from toolz import memoize
import numpy as np

def old_sort_tree(mt, state):
    """Breaks symmetry between belief states.
    
    This is done by enforcing that the knowldge about states at each
    depth be sorted by [0, 1, UNKNOWN]
    """
    state = list(state)
    for i in range(len(mt.tree) - 1, -1, -1):
        if not mt.tree[i]:
            continue  # 
        c1, c2 = mt.tree[i]
        if not (mt.subtree(state, c1) <= mt.subtree(state, c2)):
            diff = c2 - c1
            for j in range(c1, c2):
                state[j], state[j+diff] = state[j+diff], state[j]
    return tuple(state)

def sort_tree(tree):
    def sort_children(children):
        if not children:
            return children
        return tuple(sorted(map(sort_tree, children)))
    
    r, children = tree
    return(r, sort_children(children))


def solve(mdp, hash_state=sort_tree, actions=None):
    """Returns Q, V, pi, and computation data for an mdp."""
    if actions is None:
        actions = mdp.actions
    # if hash_state is None:
    #     hash_state = lambda s: sort_tree(mt, s)
    info = {  # track number of times each function is called
        'q': 0,
        'v': 0
    }

    if hash_state is None:
        hash_key = None
    else:
        def hash_key(args, kwargs):
            return mdp.term_state if args[0] is mdp.term_state else hash_state(args[0])
    
    @memoize
    def Q(s, a):
        info['q'] += 1
        return sum(p * (r + V(s1)) for p, s1, r in mdp.results(s, a))
        
    @memoize(key=hash_key)
    def V(s):
        info['v'] += 1
        return max((Q(s, a) for a in actions(s)), default=0)
    
    @memoize
    def pi(s):
        return max(actions(s), key=lambda a: Q(s, a))
    
    return Q, V, pi, info


def main():
    from meta import MouselabEnv
    from distributions import Categorical

    env = MouselabEnv(2, 2, reward=Categorical([0, 1]))
    Q, V, pi, info = solve(env)
    V(env._state)

if __name__ == '__main__':
    main()

