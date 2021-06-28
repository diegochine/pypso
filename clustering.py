import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pso.gbest import GBestPSO


def assign_clusters(X, clusters):
    distance_matrix = np.empty((X.shape[0], n_clusters))
    for i, point in enumerate(X):
        cp = np.reshape(point, (1, -1)).copy()
        distance_matrix[i, :] = np.sum((clusters - cp) ** 2, axis=1)
    labels = np.argmin(distance_matrix, axis=1)
    return labels


def calinski_harabasz_score(X, labels):
    # see https://www.researchgate.net/publication/233096619_A_Dendrite_Method_for_Cluster_Analysis
    # and https://www.journaljamcs.com/index.php/JAMCS/article/view/24229
    intra_dispersion = 0.
    extra_dispersion = 0.
    mean_data = np.mean(X, axis=0)
    for k in range(n_clusters):
        cluster = X[labels == k]
        if cluster.shape != (0, 2):
            mean_cluster = np.mean(cluster, axis=0)
            extra_dispersion += cluster.shape[0] * np.sum((mean_cluster - mean_data) ** 2)
            intra_dispersion += np.sum((cluster - mean_cluster) ** 2)
    if extra_dispersion == 0:
        return 1.
    else:
        n_samples = X.shape[0]
        n_labels = np.unique(labels).size
        return extra_dispersion * (n_samples - n_labels) / (intra_dispersion * (n_labels - 1))


def f(pos):
    """ Overhead function to perform clustering using the swarm"""
    labels = assign_clusters(X, pos)
    penalty = 0 if np.unique(labels).size == n_clusters else 1000
    return - calinski_harabasz_score(X, labels) + penalty


def animate(cluster_info, points, clusters_plots, centroids_plots, iter_text):
    (centroids, labels, iteration) = cluster_info
    print('*' * 10)
    print(f'i: {iteration}')
    for k in range(centroids.shape[0]):
        new_assignment = points[labels == k, :]
        print(f'k: {k}, pts: {new_assignment.shape[0]}')
        clusters_plots[k].set_data(new_assignment[:, 0], new_assignment[:, 1])
        centroids_plots[k].set_data(centroids[k, 0], centroids[k, 1])
    iter_text.set_text(f'ITERATION: {iteration:d}')
    if iteration == 0:
        time.sleep(1.5)


def animate_clusters(swarm, points):
    def generator():
        for i, centroids in enumerate(cluster_history):
            labels = assign_clusters(points, centroids)
            yield centroids, labels, i

    cluster_history = swarm.gbest_history
    n_clusters = cluster_history[0].shape[0]
    clusters_plots = []
    centroids_plots = []
    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    fig, ax = plt.subplots()
    for k in range(n_clusters):
        cluster_plot, = ax.plot([], [], ls='None', marker='o', ms=5,
                                markeredgecolor='k', markerfacecolor=colors[k], label=f'cluster {k}')
        clusters_plots.append(cluster_plot)

        centroid_plot, = ax.plot([], [], ls='None', marker='*', ms=15,
                                 markeredgecolor='k', markerfacecolor=colors[k])
        centroids_plots.append(centroid_plot)
    iter_text = ax.annotate('', xy=(0.01, 0.01), xycoords='axes fraction', fontsize='x-large')
    pad = 0.1
    ax.set_xlim(points[:, 0].min() - pad, points[:, 0].max() + pad)
    ax.set_ylim(points[:, 1].min() - pad, points[:, 1].max() + pad)
    ax.set_title('PSO clustering animation')
    anim = animation.FuncAnimation(fig, animate, frames=generator,
                                   fargs=(points, clusters_plots, centroids_plots, iter_text),
                                   repeat=True, interval=500)
    plt.legend()
    plt.show()
    return anim


if __name__ == '__main__':
    X = np.load('data/cluster_data.npy')
    y = np.load('data/cluster_labels.npy')
    bounds = (np.min(X, axis=0), np.max(X, axis=0))
    hyparams = {'c1': 2.1, 'c2': 2.1}
    n_clusters = 5
    swarm = GBestPSO(100, (n_clusters, 2), bounds=bounds, hyparams=hyparams, verbose=True)
    c, p = swarm.minimize(f, iters=10)
    print(f'final cost: {c}')
    ani = animate_clusters(swarm, X)
