import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.datasets import make_blobs
from sklearn.metrics import calinski_harabasz_score
from pso import GBestPSO


def assign_clusters(X, clusters):
    distance_matrix = np.empty((n_samples, k))
    for i, point in enumerate(X):
        cp = np.reshape(point, (1, -1)).copy()
        distance_matrix[i, :] = np.sum((clusters - cp) ** 2, axis=1)
    labels = np.argmin(distance_matrix, axis=1)
    return labels


def f(pos):
    """ Overhead function to perform clustering using the swarm"""
    labels = assign_clusters(X, pos)
    return - calinski_harabasz_score(X, labels)


def plot_clusters(X, y, points, labels_prefix='cluster', points_name='centroids', ):
    """
    Plot a two dimensional projection of an array of labelled points
    X:      array with at least two columns
    y:      vector of labels, length as number of rows in X
    dim:    the two columns to project, inside range of X columns, e.g. (0,1)
    points: additional points to plot as 'stars'
    labels_prefix: prefix to the labels for the legend ['cluster']
    points_name:   legend name for the additional points ['centroids']
    """
    # plot the labelled (colored) dataset and the points
    labels = np.unique(y)
    for i in range(len(labels)):
        plt.scatter(X[y == labels[i], 0],
                    X[y == labels[i], 1],
                    s=10,
                    marker='s', edgecolors='k',
                    label=labels_prefix + str(labels[i]))
    plt.scatter(points[:, 0],
                points[:, 1],
                s=50,
                marker='*',
                # c=[points_color],
                label=points_name)
    plt.legend()
    plt.grid()
    plt.show()


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
    iter_text = ax.annotate('', xy=(0.01, 0.01), xycoords='axes fraction')
    pad = 0.1
    ax.set_xlim(points[:, 0].min() - pad, points[:, 0].max() + pad)
    ax.set_ylim(points[:, 1].min() - pad, points[:, 1].max() + pad)
    anim = animation.FuncAnimation(fig, animate, frames=generator,
                                   fargs=(points, clusters_plots, centroids_plots, iter_text),
                                   repeat=True, interval=500)
    plt.legend()
    plt.grid()
    plt.show()
    return anim


if __name__ == '__main__':
    samples_per_cluster = [300, 200, 500, 300, 300]
    n_samples = np.sum(samples_per_cluster)
    n_features = 2
    centers = 5
    X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=centers,
                      cluster_std=[1.0, 0.3, 2.5, 1.5, 0.5], random_state=1)
    bounds = (np.min(X, axis=0), np.max(X, axis=0))
    hyparams = {'c1': 2.1, 'c2': 2.1, 'fully_informed': False}
    for k in range(5, 7):
        swarm = GBestPSO(150, (k, n_features), bounds=bounds, verbose=True)
        c, p = swarm.minimize(f, iters=20)
        print(f'final cost: {c}')
        ani = animate_clusters(swarm, X)
