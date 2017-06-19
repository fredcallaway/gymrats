import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Normal

class BayesianRegression(object):
    """Bayesian linear regression."""
    def __init__(self, w, sigma_y=1):
        super().__init__()
        self.sigma_y = sigma_y
        D = len(w)
        X = self._X = tf.placeholder(tf.float32, [None, D])
        w = Normal(loc=tf.zeros(D), scale=tf.ones(D) * 100)
        y = Normal(loc=ed.dot(X, w), scale=sigma_y)
        y_obs = self._y_obs = tf.placeholder(tf.float32, [None])

        qw = self.qw = Normal(loc=tf.Variable(tf.random_normal([D])),
                              scale=tf.nn.softplus(tf.Variable(tf.random_normal([D]))))

        self.inference = ed.KLqp({w: qw}, data={y: y_obs})
        self.inference.initialize(n_iter=250, n_samples=5)
        tf.global_variables_initializer().run()
    
    def update(self, X, y):
        for _ in range(250):
            info_dict = self.inference.update({self._X: X, self._y_obs: y})
            self.inference.print_progress(info_dict)
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

