import numpy as np
from numpy.random.mtrand import normal, uniform
from abc import ABC, abstractmethod


class AbstractParticle(ABC):

    def __init__(self, dim, bounds=None, vmax=None):
        """ initializes the particle
            Parameters:
                dim: int or tuple
                    dimensions of the search space
                bounds: tuple of numpy arrays, default None
                    boundaries on the search space, see optimizer class
            """
        if isinstance(dim, int):  # vector space
            self.shape = (dim, 1)
        elif isinstance(dim, tuple):  # matrix space
            self.shape = dim
        else:
            raise TypeError('dim must be either int or tuple')
        if bounds is None:
            self.lb = None
            self.ub = None
            self.current_position = uniform(-1, 1, self.shape)
        else:
            self.lb, self.ub = bounds
            if self.shape[1] == 1:  # vector space
                self.current_position = np.array([uniform(self.lb[i], self.ub[i])
                                                  for i in range(self.shape[0])]).reshape(self.shape)
            else:  # matrix space
                self.current_position = np.array([[uniform(self.lb[i], self.ub[i])
                                                   for i in range(self.shape[1])]
                                                  for _ in range(self.shape[0])])

        self.velocity = np.zeros(self.shape)
        self.vmax = vmax
        self.pbest_pos = self.current_position
        self.pbest_val = np.Inf

    def __str__(self):
        return f'PBEST: {self.pbest_val:.4f}, P: {self.current_position}, V: {self.velocity}'

    def step(self, f):
        """
        evaluate fitness function (eventually updating local best)
        """
        fval = f(self.current_position)
        if isinstance(fval, np.ndarray):
            fval = fval[0]
        if fval < self.pbest_val:
            # update best found local solution
            self.pbest_val = fval
            self.pbest_pos = self.current_position.copy()
        return self.pbest_val

    def _check_velocity(self):
        """ checks whether position update may result in particle outside the bounds,
            and eventually updates velocity to avoid it.
            also checks that velocity is within the range Vmax, and eventually updates.
        """
        if self.lb is not None:
            next_pos = self.current_position + self.velocity
            out_below = self.lb > next_pos
            out_above = next_pos > self.ub
            if np.any(np.logical_or(out_below, out_above)):
                # update velocity by inverting and randomly stretching "outsiders" dimensions
                delta_below = out_below * (self.lb - next_pos) * uniform(1, 2)
                delta_above = out_above * (self.ub - next_pos) * uniform(1, 2)
                self.velocity += delta_below + delta_above
        # clip to vmax
        self.velocity = np.minimum(self.velocity, self.vmax)

    @abstractmethod
    def update(self, data, **kwargs):
        """ update parameters of the particle """
        pass


class StandardParticle(AbstractParticle):
    """ standard particle: update is performed using only personal and neighborhood bests """

    def __init__(self, dim, bounds=None, vmax=1e6):
        super().__init__(dim, bounds=bounds, vmax=vmax)

    def update(self, neighbor_best, **kwargs):
        constriction, phi1, phi2 = kwargs['constriction'], kwargs['phi1'], kwargs['phi2']
        personal_change = uniform(0, phi1, self.shape) * (self.pbest_pos - self.current_position)
        neighborhood_change = uniform(0, phi2, self.shape) * (neighbor_best - self.current_position)
        self.velocity = constriction * (self.velocity + personal_change + neighborhood_change)
        self._check_velocity()
        self.current_position += self.velocity


class FullyInformedParticle(AbstractParticle):
    """ fully informed particle: update is performed using all neighbors' bests"""

    def __init__(self, dim, bounds=None, vmax=1e6):
        super().__init__(dim, bounds=bounds, vmax=vmax)

    def update(self, data, **kwargs):
        constriction, phi = kwargs['constriction'], kwargs['phi']
        k = data.shape[0]
        term = np.random.uniform(0, phi, data.shape) * (data - self.current_position)
        self.velocity = constriction * (self.velocity + np.sum(term, axis=0) / k)
        self._check_velocity()
        self.current_position += self.velocity
