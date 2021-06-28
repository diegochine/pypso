import numpy as np
from pso.utils import plot_swarm
from pso.gbest import GBestPSO
from pso.lbest import LBestPSO


def ackley(x, y=None):
    if y is None:
        x, y = x[0], x[1]
    return (- 20.0 * np.exp(-0.2 * np.sqrt(0.5 * (x ** 2 + y ** 2)))
            - np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)))
            + np.exp(1) + 20.0
            )


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


def rastrigin(x, y=None):
    if y is None:
        x, y = x[0], x[1]
    A = 10.0
    return (A * 2 +
            (x ** 2 - A * np.cos(2 * np.pi * x)) +
            (y ** 2 - A * np.cos(2 * np.pi * y))
            )


def easom(x, y=None):
    if y is None:
        x, y = x[0], x[1]
    return -1 * np.cos(x) * np.cos(y) * np.exp(-1 * ((x - np.pi) ** 2 + (y - np.pi) ** 2))


if __name__ == '__main__':
    benchmark = {
        'sphere': (sphere, (0, 0), None),
        'ackley': (ackley, (0, 0), (np.full((2, 1), -5), np.full((2, 1), 5))),
        'booth': (booth, (1, 3), (np.full((2, 1), -10), np.full((2, 1), 10))),
        'easom': (easom, (np.pi, np.pi), (np.full((2, 1), -100), np.full((2, 1), 100))),
        'matyas': (matyas, (0, 0), (np.full((2, 1), -10), np.full((2, 1), 10))),
        'rastrigin': (rastrigin, (0, 0), (np.full((2, 1), -5.12), np.full((2, 1), 5.12))),
        'beale': (beale, (3, 0.5), (np.full((2, 1), -4.5), np.full((2, 1), 4.5)))
    }
    n_particles = 200
    hyparams = {'c1': 2.05, 'c2': 2.05,
                'k': 10, 'dynamic': True, 'kfun': lambda _, k: k + 1,
                'fully_informed': True}
    visualize = True
    tol = 1e-03
    max_iter = 50

    for fname in benchmark:
        print(f'Minimizing benchmark function: {fname}')
        fobj, true_min_pos, bounds = benchmark[fname]
        swarm = GBestPSO(n_particles, 2, hyparams=hyparams, bounds=bounds, verbose=True)
        found_min_val, found_min_pos = swarm.minimize(fobj, max_iter)
        with np.printoptions(precision=3, suppress=True):  # prettier printing
            true_min_pos = np.array(true_min_pos, dtype=np.float64).reshape(2, 1)
            true_min_val = fobj(true_min_pos)[0]
            dist = np.sum((found_min_pos - true_min_pos) ** 2)
            print(f'Found minimum: {found_min_pos.reshape(-1)} with value {found_min_val:.3f}')
            print(f'True minimum : {true_min_pos.reshape(-1)} with value {true_min_val:.3f}')
            print(f'Distance (euclidean): {dist:.3f}')
            convergence = np.argwhere(np.abs(np.array(swarm.cost_history) - true_min_val) < tol)
            if np.any(convergence):
                convergence = convergence[0, 0]
                print(f'Convergence speed (iterations before dist < {tol}): {convergence:3d}/{max_iter:3d}')
            else:
                print(f"Did not converge with tolerance {tol}")
            print()
        if visualize:
            plot_swarm(swarm, fobj, bounds=bounds, plot_surf=True, plot_proj=False)
