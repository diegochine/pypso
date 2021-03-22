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
    benchmark = [(sphere, (0, 0)), (beale, (3, 0.5)),
                 (booth, (1, 3)), (matyas, (0, 0))]
    hyparams = {'c1': 2.1, 'c2': 2.1, 'k': 50, 'fully_informed': True}
    for fun, gmin in benchmark:
        print(f'Minimizing benchmark function: {fun.__name__}')
        swarm = GBestPSO(500, 2, hyparams=hyparams)
        v, p = swarm.minimize(fun, 50)
        with np.printoptions(precision=3, suppress=True):  # prettier printing
            print(f'Found minimum at p:  {p} with value v = {v:.3f}')
            print(f'True global minimum: {gmin}\n')
        plot_swarm(swarm, fun, plot_surf=True, plot_proj=False)
