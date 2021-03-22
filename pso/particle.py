import numpy as np
from numpy.random.mtrand import normal, uniform
from abc import ABC, abstractmethod


class AbstractParticle(ABC):

    def __init__(self, dim, alpha=1):
        self.current_position = normal(0, 1, dim)
        self.shape = dim
        self.velocity = np.zeros(dim)  # normal(0, 1, dim)
        self.alpha = alpha
        self.pbest_pos = self.current_position
        self.pbest_val = np.Inf

    def __str__(self):
        return f'PBEST: {self.pbest_val:.4f}, P: {self.current_position}, V: {self.velocity}'

    def step(self, f):
        """
        evaluate fitness function (eventually updating local best)
        """
        fval = f(self.current_position)
        if fval < self.pbest_val:
            # update best found local solution
            self.pbest_val = fval
            self.pbest_pos = self.current_position
        return self.pbest_val

    @abstractmethod
    def update(self, data, **kwargs):
        pass


class StandardParticle(AbstractParticle):

    def __init__(self, dim):
        super().__init__(dim)

    def update(self, global_best, **kwargs):
        constriction, phi1, phi2 = kwargs['constiction'], kwargs['phi1'], kwargs['phi2']
        local_change = uniform(0, phi1, self.shape) * (self.pbest_pos - self.current_position)
        global_change = uniform(0, phi2, self.shape) * (global_best - self.current_position)
        self.velocity = constriction * (self.velocity + local_change + global_change)
        self.current_position += self.alpha * self.velocity


class FullyInformedParticle(AbstractParticle):

    def __init__(self, dim):
        super().__init__(dim)

    def update(self, data, **kwargs):
        constriction, phi = kwargs['constriction'], kwargs['phi']
        k = data.shape[0]
        term = np.random.uniform(0, phi, data.shape) * (data - self.current_position)
        self.velocity = constriction * (self.velocity + np.sum(term, axis=0)/k)
        self.current_position += self.velocity
