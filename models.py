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


class BayesianRegression(FunctionApproximator):
    """Bayesian linear regression."""
    def __init__(self, w, sigma_y=1.0, sigma_w=100):
        super().__init__()
        self.sigma_y = float(sigma_y)
        D = len(w)
        X = self._X = tf.placeholder(tf.float32, [None, D])
        w = Normal(loc=tf.zeros(D), scale=tf.ones(D) * sigma_w)
        y = Normal(loc=ed.dot(X, w), scale=sigma_y)
        y_obs = self._y_obs = tf.placeholder(tf.float32, [None])

        qw = self.qw = Normal(loc=tf.Variable(tf.random_normal([D]) * 0.01),
                              scale=tf.nn.softplus(tf.Variable(tf.random_normal([D]) * 0.01)))

        self.inference = ed.KLqp({w: qw}, data={y: y_obs})
        self.inference.initialize(n_iter=100, n_samples=5)
        tf.global_variables_initializer().run()
        self.w = self.qw.loc.eval()
        self.sigma_w = self.qw.scale.eval()
    
    def update(self, X, y, n_iter=1):
        for _ in range(n_iter):
            info_dict = self.inference.update({self._X: X, self._y_obs: y})
            # self.inference.print_progress(info_dict)
        self.w = self.qw.loc.eval()
        self.sigma_w = self.qw.scale.eval()
        # n_sample = 100
        # self._ws = self.qw.sample(n_sample).eval()
        # self._bs = self.qb.sample(n_sample).eval()

    def predict(self, x):
        # return np.sum(x * self._ws, axis=1) + self._bs
        mean = self.w @ x
        var = self.sigma_y ** 2 + (x * self.sigma_w * x).sum()
        return mean, var



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
