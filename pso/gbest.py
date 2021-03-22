import numpy as np
from pso.optimizer import AbstractOptimizer


class GBestPSO(AbstractOptimizer):

    """Global best PSO"""

    def __init__(self, n_particles, dimensions, hyparams=None):
        super().__init__(n_particles, dimensions, hyparams, 'gbest', 'logs/gbest.log')

    def __str__(self):
        return '\n'.join([f'{particle}' for particle in self.particles])

    def minimize(self, f, iters=100):
        for iteration in range(iters):
            local_bests = np.array([particle.step(f) for particle in self.particles])
            best_particle = self.particles[np.argmin(local_bests)]
            if best_particle.pbest_val < self.gbest_value:
                self.logger.info('UPDATING GBEST')
                self.gbest_position = best_particle.pbest_pos.copy()
                self.gbest_value = best_particle.pbest_val

            if self.fully_informed:
                pbest_matrix = np.array([p.pbest_pos for p in self.particles])
                for particle in self.particles:
                    particle.update(pbest_matrix, constriction=self.constriction, phi=self.phi)
            else:
                for particle in self.particles:
                    particle.update(self.gbest_position, constriction=self.constriction, phi1=self.c1, phi2=self.c2)

            self._log(iteration, iters)
            self._update_history(f)

        return self.gbest_value, self.gbest_position
