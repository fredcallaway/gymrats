from abc import ABC, abstractmethod
import edward as ed
import numpy as np
import tensorflow as tf
from edward.models import Normal

class FunctionApproximator(object):
    """Approximates functions."""
    
    @abstractmethod  
    def update(self, X, y):
        pass
    
    @abstractmethod
    def predict(self, x):
        pass


def positive_variable(shape=(), init=0):
    return tf.sqrt(tf.exp(variable(shape)))

def variable(shape=(), init=0):
    return tf.Variable(tf.random_normal(shape) * .01)
    # return tf.Variable(tf.zeros(shape))

def data(shape, dtype='float32'):
    return tf.placeholder(dtype, shape)


class BayesianRegression(FunctionApproximator):
    """Bayesian linear regression."""
    def __init__(self, w_prior, sigma_w=100.):
        super().__init__()
        w_prior = np.atleast_2d(w_prior)
        shape = w_prior.shape

        # Linear regression model.
        self._X = data((None, shape[0]))
        w = Normal(loc=tf.zeros(shape), scale=sigma_w)
        # Observation noise is optimized as a point estimate
        # so it doesn't have an associated distribution.
        self._sigma_y = positive_variable()
        y = Normal(loc=tf.matmul(self._X, w), scale=self._sigma_y)
        self._y_obs = tf.placeholder(tf.float32, [None, shape[1]])

        # Varitional inference.
        qw = self.qw = Normal(loc=variable(shape), scale=positive_variable(shape))
        self.inference = ed.KLqp({w: qw}, data={y: self._y_obs})
        self.inference.initialize(n_iter=100, n_samples=5)
        tf.global_variables_initializer().run()
        self.w = self.qw.loc.eval()
        self.sigma_w = self.qw.scale.eval()
        self._sigma_w_T = self.sigma_w.T
    
    def update(self, X, y, n_iter=1):
        for _ in range(n_iter):
            info_dict = self.inference.update({self._X: X, self._y_obs: y})
            # self.inference.print_progress(info_dict)
        self.w = self.qw.loc.eval()
        self.sigma_w = self.qw.scale.eval()
        self._sigma_w_T = self.sigma_w.T

    def predict(self, x, return_var=False):
        mean = x @ self.w
        if return_var:
            var = (x * self._sigma_w_T * x).sum(1)
            return mean, var
        else:
            return mean

class BayesQ(FunctionApproximator):
    """Bayesian linear regression."""
    def __init__(self, w_prior, sigma_w=100., n_iter=100):
        super().__init__()
        w_prior = np.atleast_2d(w_prior)
        shape = w_prior.shape

        # States and (one-hot encoded) actions are input.
        self._states = data([None, shape[0]])
        self._actions = data([None, shape[1]])
        self._qs = data([None])


        # Linear regression weights.
        w = Normal(loc=tf.zeros(shape), scale=sigma_w)

        # Observation noise resulting from non-determinism in the environment
        # and/or policy. This parameter is optimized as a point estimate.
        self.sigma_y = positive_variable()

        # Observed Q value depends on Q values of all actions and the action taken.
        # Q = Normal(loc=tf.matmul(self._states, w), scale=self.sigma_y)
        Q = tf.matmul(self._states, w)
        qs = Normal(loc=tf.reduce_sum(Q * self._actions, 1),
                    scale=self.sigma_y)

        # Varitional inference.
        self.qw = Normal(loc=variable(shape), scale=positive_variable(shape))
        self.inference = ed.KLqp({w: self.qw}, data={qs: self._qs})
        self.inference.initialize(n_iter=n_iter, n_samples=5)
        tf.global_variables_initializer().run()
        
        # Inferred parameters.
        self.w = self.qw.loc.eval()
        self.sigma_w = self.qw.scale.eval()
        self._sigma_w_T = self.sigma_w.T

    def update(self, states, actions, qs, n_iter=1):
        for _ in range(n_iter):
            update = {self._states: states, self._actions: actions, self._qs: qs}
            info_dict = self.inference.update(update)
            # self.inference.print_progress(info_dict)
        self.w = self.qw.loc.eval()
        self.sigma_w = self.qw.scale.eval()
        self._sigma_w_T = self.sigma_w.T

    def predict(self, x, return_var=False):
        x = np.atleast_2d(x)
        mean = x @ self.w
        if return_var:
            var = (x * self._sigma_w_T * x).sum(1)
            return mean, var
        else:
            return mean
      

# class BayesQ(FunctionApproximator):
#     """Bayesian linear regression."""
#     def __init__(self, w_prior, sigma_w=100.):
#         super().__init__()
#         w_prior = np.atleast_2d(w_prior)
#         state_size, n_action = w_prior.shape

#         # Observed data.
#         self._states = data([None, state_size])
#         self._qs = data([None])

#         # # Linear regression weights.
#         # w = Normal(loc=tf.zeros(shape), scale=sigma_w)

#         # # Each action has separate variance due to observation noise resulting
#         # # from non-determinism in the environment and/or policy.
#         # # This parameter is optimized as a point estimate.
#         # self.sigma_y = positive_variable()

#         # # Observed Q value depends on Q values of all actions and the action taken.
#         # # Q = Normal(loc=tf.matmul(self._states, w), scale=self.sigma_y)
#         # Q = tf.matmul(self._states, w)
#         # qs = Normal(loc=tf.reduce_sum(Q * self._actions, 1),
#         #             scale=self.sigma_y)


#         # Linear regression weights.
#         weights = [Normal(loc=tf.zeros(state_size), scale=sigma_w)
#                    for _ in range(n_action)]
#         # Each action has separate variance due to observation noise resulting
#         # from non-determinism in the environment and/or policy.
#         # This parameter is optimized as a point estimate.
#         self.sigma_y = positive_variable()

#         # qs = Normal(loc=w[self._actions])
#         Q = [ed.dot(self._states, w) for w in weights]        

#         # Observed Q value depends on Q values of all actions and the action taken.
#         # Q = Normal(loc=tf.matmul(self._states, w), scale=self.sigma_y)
#         Q = tf.matmul(self._states, w)
#         qs = Normal(loc=tf.reduce_sum(Q * self._actions, 1),
#                     scale=self.sigma_y)

#         # Varitional inference.
#         self.qw = [Normal(loc=variable(state_size), scale=positive_variable([]))
#                    for _ in range(n_action)]
#         self.inference = ed.KLqp({w: self.qw}, data={qs: self._qs})
#         self.inference.initialize(n_iter=100, n_samples=5)
#         # tf.global_variables_initializer().run()
        
#         # Inferred parameters.
#         self.w = self.qw.loc.eval()
#         self.sigma_w = self.qw.scale.eval()
#         self._sigma_w_T = self.sigma_w.T
    
#     def update(self, states, actions, qs, n_iter=1):
#         for _ in range(n_iter):
#             update = {self._states: states, self._actions: actions, self._qs: qs}
#             info_dict = self.inference.update(update)
#             self.inference.print_progress(info_dict)
#         self.w = self.qw.loc.eval()
#         self.sigma_w = self.qw.scale.eval()
#         self._sigma_w_T = self.sigma_w.T

#     def predict(self, x, return_var=False):
#         mean = x @ self.w
#         if return_var:
#             var = (x * self._sigma_w_T * x).sum(1)
#             return mean, var
#         else:
#             return mean

class Network(object):
    """docstring for Network"""
    def __init__(self, shape):
        super().__init__()
        self.shape = shape
        model = Sequential()
        model.add(Dense(24, input_dim=shape[0], activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(24, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(shape[1], activation='linear',
                        kernel_initializer='he_uniform'))
        self.net.compile(optimizer='rmsprop', loss='mse')
      

class LinearSGD(object):
    """Learns a linear approximation by SGD."""
    def __init__(self, shape, learn_rate=.1):
        self.shape = shape
        self.learn_rate = learn_rate
        self.theta = np.random.random(self.shape)

    def update(self, x, y):
        yhat = x @ self.theta
        error = y - yhat
        self.theta += self.learn_rate * np.outer(x, error)

    def predict(self, x):
        return x @ self.theta
