import numpy as np
from logger import setup_logger
from pso.optimizer import AbstractOptimizer


class GBestPSO(AbstractOptimizer):

    """Global best PSO"""

    def __init__(self, n_particles, dimensions, hyparams=None):
        super().__init__(n_particles, dimensions, hyparams)
        self.logger = setup_logger('fipso', 'logs/gbest.txt')

    def __str__(self):
        return '\n'.join([f'{particle}' for particle in self.particles])

    def minimize(self, f, iters=100):
        for iter in range(iters):
            self.logger.info(f'ITER {iter:3d}/{iters:3d}')
            local_bests = np.array([particle.step(f) for particle in self.particles])
            best_particle = self.particles[np.argmin(local_bests)]
            if best_particle.pbest_val < self.gbest_value:
                self.logger.info('UPDATING GBEST')
                self.gbest_position = best_particle.pbest_pos.copy()
                self.gbest_value = best_particle.pbest_val

            for particle in self.particles:
                particle.update(self.gbest_position, self.constriction, self.c1, self.c2)
            self.logger.info(f'GLOBAL BEST POSITION: {self.gbest_position} '
                             f'WITH VALUE {self.gbest_value:3f}')
            self.logger.info(f'SWARM STATUS')
            self.logger.info('\n' + '\n'.join([f'P{i:03d}: {p}' for i, p in enumerate(self.particles)]))
            #print(f'VALUE: {self.gbest_value:.4f}, POS: {self.gbest_position}')
            #print(f'SWARM CENTROID: {np.mean([p.current_position for p in self.particles])}')

            self.particles_history.append(
                np.array([np.append(p.current_position, f(p.current_position))
                          for p in self.particles]))
            self.cost_history.append(self.gbest_value)
        return self.gbest_value, self.gbest_position