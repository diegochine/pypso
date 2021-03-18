import numpy as np
from abc import ABC, abstractmethod
from pso.particle import Particle


class AbstractOptimizer(ABC):
    """ Generic base classe for all PSO optimizers """

    DEFAULT_HYPARAMS = {'c1': 2.05, 'c2': 2.05, 'w': 1, 'k': 3}

    def __init__(self, n_particles, dimensions, hyparams):
        # particles status
        self.particles = [Particle(dimensions) for _ in range(n_particles)]
        self.position_matrix = np.array([p.current_position for p in self.particles])
        self.velocity_matrix = np.array([p.velocity for p in self.particles])
        # hyperparameters setup
        if hyparams is None or not isinstance(hyparams, dict):
            hyparams = self.DEFAULT_HYPARAMS
        else:
            hyparams = {**self.DEFAULT_HYPARAMS, **hyparams}
        self.c1 = hyparams['c1']
        self.c2 = hyparams['c2']
        self.phi = self.c1 + self.c2
        self.constriction = 2 / (self.phi - 2 + np.sqrt(self.phi ** 2 - 4 * self.phi))
        self.k = hyparams['k']
        # histories
        self.gbest_value = np.inf
        self.gbest_position = None
        self.particles_history = []
        self.cost_history = []

    @abstractmethod
    def minimize(self, f, iters):
        pass

    def update_position_matrix(self):
        self.position_matrix = np.array([p.current_position for p in self.particles])

    def update_velocity_matrix(self):
        self.velocity_matrix = np.array([p.velocity for p in self.particles])

    def update_history(self):
        pass
