import numpy as np
import matplotlib.pyplot as plt
from plot import generate_data, plot_swarm
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


swarm = LBestPSO(100, 2, hyparams={'k': 2})
v, p = swarm.minimize(sphere, 100)
print(f'v = {v}, p = {p}')
# plt.plot(swarm.cost_history)
plot_swarm(swarm.particles_history, fobj=sphere, lowlim=-10, highlim=10)
