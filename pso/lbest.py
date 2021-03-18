import numpy as np
from scipy.spatial import KDTree

from logger import setup_logger
from pso.optimizer import AbstractOptimizer


class LBestPSO(AbstractOptimizer):
    """Original pso algorithm (local-best)"""

    def __init__(self, n_particles, dimensions, hyparams=None):
        super().__init__(n_particles, dimensions, hyparams)
        self.logger = setup_logger('pso', 'logs/lbest.txt')

    def minimize(self, f, iters=100):
        for iter in range(iters):
            self.logger.info(f'ITER {iter:3d}/{iters:3d}')
            neighbors = KDTree(self.position_matrix)
            for i, particle in enumerate(self.particles):
                particle.step(f)
                _, neighbors_idx = neighbors.query(self.position_matrix[i], k=self.k)
                neighbors_bests = np.array([self.particles[idx].pbest_val for idx in neighbors_idx])
                best_neighbor = self.particles[np.argmin(neighbors_bests)]
                particle.update(best_neighbor.pbest_pos, self.constriction, self.c1, self.c2)
                if best_neighbor.pbest_val < self.gbest_value:
                    self.gbest_position = best_neighbor.pbest_pos.copy()
                    self.gbest_value = best_neighbor.pbest_val

            self.logger.info(f'GLOBAL BEST POSITION: {self.gbest_position} '
                             f'WITH VALUE {self.gbest_value:3f}')
            self.logger.info(f'SWARM STATUS')
            self.logger.info('\n' + '\n'.join([f'P{i:03d}: {p}' for i, p in enumerate(self.particles)]))
            print(f'VALUE: {self.gbest_value:.4f}, POS: {self.gbest_position}')
            print(f'SWARM CENTROID: {np.mean([p.current_position for p in self.particles])}')

            self.particles_history.append(
                np.array([np.append(p.current_position, f(p.current_position[0], p.current_position[1]))
                          for p in self.particles]))
            self.cost_history.append(self.gbest_value)
        return self.gbest_value, self.gbest_position
