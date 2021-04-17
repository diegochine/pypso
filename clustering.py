import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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


def plot_clusters(X, y, points,
                  labels_prefix='cluster',
                  points_name='centroids',
                  colors=cm.tab10,
                  points_color=cm.tab10(10)  # by default the last of the map (to be improved)
                  ):
    """
    Plot a two dimensional projection of an array of labelled points
    X:      array with at least two columns
    y:      vector of labels, length as number of rows in X
    dim:    the two columns to project, inside range of X columns, e.g. (0,1)
    points: additional points to plot as 'stars'
    labels_prefix: prefix to the labels for the legend ['cluster']
    points_name:   legend name for the additional points ['centroids']
    colors: a color map
    points_color: the color for the points
    """
    # plot the labelled (colored) dataset and the points
    labels = np.unique(y)
    for i in range(len(labels)):
        color = colors(i / len(labels))  # choose a color from the map
        plt.scatter(X[y == labels[i], 0],
                    X[y == labels[i], 1],
                    s=10,
                    c=[color],  # scatter requires a sequence of colors
                    marker='s',
                    label=labels_prefix + str(labels[i]))
    plt.scatter(points[:, 0],
                points[:, 1],
                s=50,
                marker='*',
                c=[points_color],
                label=points_name)
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    n_samples = 1500
    n_features = 2
    centers = 5
    X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=centers)
    bounds = (np.min(X, axis=0), np.max(X, axis=0))
    hyparams = {'c1': 2.1, 'c2': 2.1, 'fully_informed': False}
    for k in range(3, 7):
        swarm = GBestPSO(50, (k, n_features), bounds=bounds, verbose=True)
        c, p = swarm.minimize(f, iters=10)
        print(f'final cost: {c}')
        plot_clusters(X, assign_clusters(X, p), p)
