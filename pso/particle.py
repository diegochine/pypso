import numpy as np
from numpy.random.mtrand import normal, uniform


class Particle:

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
        # evaluate fitness function
        fval = f(self.current_position)
        if fval < self.pbest_val:
            # update best found local solution
            self.pbest_val = fval
            self.pbest_pos = self.current_position
        return self.pbest_val

    def update(self, global_best, constriction, phi1, phi2):
        local_change = uniform(0, phi1, self.shape) * (self.pbest_pos - self.current_position)
        global_change = uniform(0, phi2, self.shape) * (global_best - self.current_position)
        self.velocity = constriction * (self.velocity + local_change + global_change)
        self.current_position += self.alpha * self.velocity
