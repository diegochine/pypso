import numpy as np
from abc import ABC, abstractmethod

from pso.utils.logger import setup_logger
from pso.particle import StandardParticle, FullyInformedParticle


class AbstractOptimizer(ABC):
    """ Generic base class for all PSO optimizers """

    DEFAULT_HYPARAMS = {'c1': 2.05,
                        'c2': 2.05,
                        'k': 10,
                        'dynamic': True,
                        'kfun': lambda _, k: k + 1,
                        'fully_informed': False}

    def __init__(self, n_particles, dimensions, hyparams, logger_name, log_path,
                 bounds=None, limit_vmax=True, verbose=False):
        """
        Initialize the optimizer
        Parameters:
            n_particles: int
                number of particles in the swarms
            dimensions: int or tuple
                dimensions in the search space
            hyparams: dict with the following (string) keys:
                c1: float, cognitive weight
                c2: float, social weight
                k: int, size of neighboorhood (local-best only)
                dynamic: bool, if True then k evolves over time (must provide kfun) (local-best only)
                kfun: function to compute next value of k (local-best only)
                fully_informed: bool, if True then use fully informed particles
            logger_name: string
                name of the logger
            log_path: string
                complete path to log file (eg "logs/opt.log")
            bounds: tuple of numpy arrays, default None
                search space bounds, tuple of size 2 where first array is lower bound
                and second array is upper bound.
            limit_vmax: bool, default True
                if True, limit maximum velocity of particles to Xmax (dynamic range of the variables) in each dimension
            verbose: bool, default None
                log verbosity
        """
        print('Initializing swarm')
        # default hyparameters
        if hyparams is None or not isinstance(hyparams, dict):
            hyparams = self.DEFAULT_HYPARAMS
        else:
            hyparams = {**self.DEFAULT_HYPARAMS, **hyparams}

        if bounds is not None and limit_vmax:
            self.vmax = np.array([np.abs(bmax - bmin) for bmin, bmax in zip(bounds[0], bounds[1])])
        else:
            self.vmax = np.array([[1e6] for _ in range(dimensions)])

        # particles status
        if hyparams['fully_informed']:
            self.particles = [FullyInformedParticle(dimensions, bounds=bounds, vmax=self.vmax)
                              for _ in range(n_particles)]
        else:
            self.particles = [StandardParticle(dimensions, bounds=bounds, vmax=self.vmax)
                              for _ in range(n_particles)]
        self.bounds = bounds
        self.position_matrix = np.array([p.current_position for p in self.particles]).reshape(n_particles, -1)
        self.velocity_matrix = np.array([p.velocity for p in self.particles]).reshape(n_particles, -1)

        # hyperparameters setup
        self.c1 = hyparams['c1']
        self.c2 = hyparams['c2']
        self.phi = self.c1 + self.c2
        self.constriction = 2 / (self.phi - 2 + np.sqrt(self.phi ** 2 - 4 * self.phi))
        self.k = hyparams['k']
        self.fully_informed = hyparams['fully_informed']
        self.dynamic = hyparams['dynamic']
        if self.dynamic:
            self.kfun = hyparams['kfun']

        # histories
        self.gbest_value = np.inf
        self.gbest_position = None
        self.particles_history = []
        self.cost_history = []
        self.gbest_history = []
        self.avg_pbest_history = []

        self.logger = setup_logger(logger_name, log_path)
        self.verbose = verbose

    def __str__(self):
        return '\n'.join([f'{particle}' for particle in self.particles])

    @abstractmethod
    def minimize(self, f, iters):
        """
        minimize function for given number of iterations.
        Parameters:
            f: function
                function to be minimized. must accept a single parameter x,
                a numpy array of same size as number of search space dimensions
            iters: int
                total number of iterations
        """
        pass

    def _log(self, cur_it, tot_it):
        """ log after each iteration """
        self.logger.info(f'ITER {cur_it:3d}/{tot_it:3d}')
        self.logger.info(f'GLOBAL BEST POSITION: {self.gbest_position} '
                         f'WITH VALUE {self.gbest_value:3f}')
        if self.verbose:
            self.logger.info(f'SWARM STATUS')
            self.logger.info('\n' + '\n'.join([f'P{i:03d}: {p}' for i, p in enumerate(self.particles)]))
        if cur_it % 10 == 0:
            print(f'Iteration {cur_it:>3}/{tot_it:>3}, global best: {self.gbest_value:.3f}')

    def _update_history(self, f):
        """ update history after each iteration """
        self.particles_history.append(
            np.array([np.append(p.current_position, f(p.current_position))
                      for p in self.particles]))
        self.cost_history.append(self.gbest_value)
        self.gbest_history.append(self.gbest_position)
        self.avg_pbest_history.append(np.mean([p.pbest_val for p in self.particles]))

    def _update_position_matrix(self):
        self.position_matrix = np.array([p.current_position for p in self.particles]).reshape(self.position_matrix.shape)

    def _update_velocity_matrix(self):
        self.velocity_matrix = np.array([p.velocity for p in self.particles]).reshape(self.velocity_matrix.shape)

    def reset(self):
        """ resets histories and variables from the previous optimization """
        self.particles_history = []
        self.cost_history = []
        self.gbest_history = []
        self.gbest_value = np.Inf
        self.gbest_position = None
