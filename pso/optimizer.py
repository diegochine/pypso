import numpy as np
from abc import ABC, abstractmethod

from logger import setup_logger
from pso.particle import StandardParticle, FullyInformedParticle


class AbstractOptimizer(ABC):
    """ Generic base class for all PSO optimizers """

    DEFAULT_HYPARAMS = {'c1': 2.05, 'c2': 2.05, 'k': 10, 'fully_informed': False}

    def __init__(self, n_particles, dimensions, hyparams, logger_name, log_path, bounds=None, verbose=False):
        # default hyperparameters
        if hyparams is None or not isinstance(hyparams, dict):
            hyparams = self.DEFAULT_HYPARAMS
        else:
            hyparams = {**self.DEFAULT_HYPARAMS, **hyparams}
        # particles status
        if hyparams['fully_informed']:
            self.particles = [FullyInformedParticle(dimensions, bounds) for _ in range(n_particles)]
        else:
            self.particles = [StandardParticle(dimensions, bounds) for _ in range(n_particles)]
        self.bounds = bounds
        self.position_matrix = np.array([p.current_position for p in self.particles])
        self.velocity_matrix = np.array([p.velocity for p in self.particles])
        # hyperparameters setup
        self.c1 = hyparams['c1']
        self.c2 = hyparams['c2']
        self.phi = self.c1 + self.c2
        self.constriction = 2 / (self.phi - 2 + np.sqrt(self.phi ** 2 - 4 * self.phi))
        self.k = hyparams['k']
        self.fully_informed = hyparams['fully_informed']

        # histories
        self.gbest_value = np.inf
        self.gbest_position = None
        self.particles_history = []
        self.cost_history = []
        self.gbest_history = []

        self.logger = setup_logger(logger_name, log_path)
        self.verbose = verbose

    @abstractmethod
    def minimize(self, f, iters):
        pass

    def _log(self, cur_it, tot_it):
        self.logger.info(f'ITER {cur_it:3d}/{tot_it:3d}')
        self.logger.info(f'GLOBAL BEST POSITION: {self.gbest_position} '
                         f'WITH VALUE {self.gbest_value:3f}')
        if self.verbose:
            self.logger.info(f'SWARM STATUS')
            self.logger.info('\n' + '\n'.join([f'P{i:03d}: {p}' for i, p in enumerate(self.particles)]))

    def _update_history(self, f):
        self.particles_history.append(
            np.array([np.append(p.current_position, f(p.current_position))
                      for p in self.particles]))
        self.cost_history.append(self.gbest_value)
        self.gbest_history.append(self.gbest_position)

    def _update_position_matrix(self):
        self.position_matrix = np.array([p.current_position for p in self.particles])

    def _update_velocity_matrix(self):
        self.velocity_matrix = np.array([p.velocity for p in self.particles])

    def reset(self):
        self.particles_history = []
        self.cost_history = []
        self.gbest_history = []
        self.gbest_value = np.Inf
        self.gbest_position = None
