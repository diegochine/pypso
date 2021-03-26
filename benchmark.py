import numpy as np
from plot import plot_swarm
from pso import GBestPSO, LBestPSO


def sphere(x, y=None):
    if y is None:
        x, y = x[0], x[1]
    return x ** 2 + y ** 2


def beale(x, y=None):
    if y is None:
        x, y = x[0], x[1]
    return (1.5 - x + x * y) ** 2 + (2.25 - x + x * (y ** 2)) ** 2 + (2.625 - x + x * (y ** 3)) ** 2


def booth(x, y=None):
    if y is None:
        x, y = x[0], x[1]
    return (x + 2 * y - 7) ** 2 + (2 * x + y - 5) ** 2


def matyas(x, y=None):
    if y is None:
        x, y = x[0], x[1]
    return 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y


if __name__ == '__main__':
    benchmark = {
        'sphere': (sphere, (0, 0), None),
        'beale': (beale, (3, 0.5), (np.full(2, -4.5), np.full(2, 4.5))),
        'booth': (booth, (1, 3), (np.full(2, -10), np.full(2, 10))),
        'matyas': (matyas, (0, 0), (np.full(2, -10), np.full(2, 10)))
    }
    hyparams = {'c1': 2.1, 'c2': 2.1, 'k': 10, 'fully_informed': False}
    for fname in benchmark:
        print(f'Minimizing benchmark function: {fname}')
        fun, gmin, bounds = benchmark[fname]
        swarm = GBestPSO(100, 2, hyparams=hyparams, bounds=bounds, verbose=True)
        v, p = swarm.minimize(fun, 100)
        with np.printoptions(precision=3, suppress=True):  # prettier printing
            print(f'Found minimum at p:  {p} with value v = {v:.3f}')
            print(f'True global minimum: {gmin}\n')
        plot_swarm(swarm, fun, bounds=bounds, plot_surf=True, plot_proj=False)
