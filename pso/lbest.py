import numpy as np
from scipy.spatial import KDTree
from pso.optimizer import AbstractOptimizer


class LBestPSO(AbstractOptimizer):
    """Original pso algorithm (local-best)"""

    def __init__(self, n_particles, dimensions, hyparams=None):
        super().__init__(n_particles, dimensions, hyparams, 'lbest', 'logs/lbest.log')

    def minimize(self, f, iters=100):
        for iteration in range(iters):
            neighbors = KDTree(self.position_matrix)
            for i, particle in enumerate(self.particles):
                particle.step(f)
                _, neighbors_idx = neighbors.query(self.position_matrix[i], k=self.k)
                if self.fully_informed:
                    neighbors_pos = np.array([self.particles[idx].pbest_pos for idx in neighbors_idx])
                    particle.update(neighbors_pos, constriction=self.constriction, phi=self.phi)
                else:
                    neighbors_bests = np.array([self.particles[idx].pbest_val for idx in neighbors_idx])
                    best_neighbor = self.particles[np.argmin(neighbors_bests)]
                    particle.update(best_neighbor.pbest_pos, constriction=self.constriction, phi1=self.c1, phi2=self.c2)
                if particle.pbest_val < self.gbest_value:
                    self.gbest_position = particle.pbest_pos.copy()
                    self.gbest_value = particle.pbest_val

            self._update_position_matrix()
            self._update_velocity_matrix()

            self._log(iteration, iters)
            self._update_history(f)

        return self.gbest_value, self.gbest_position
