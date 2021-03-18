import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import log_loss, accuracy_score
from pso import GBestPSO


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=0)


def unpack(params, net_structure):
    weights = []
    biases = []
    acc = 0
    for this_layer_size, next_layer_size in zip(net_structure[:-1], net_structure[1:]):
        n_params = acc + this_layer_size * next_layer_size
        weights.append(params[acc:n_params].reshape((this_layer_size, next_layer_size)))
        biases.append(params[n_params:n_params + next_layer_size].reshape((next_layer_size,)))
        acc = n_params
    return weights, biases


def forward(x, weights, biases):
    """ Forward pass on the network """
    z = x.copy()
    for w, b in zip(weights[:-1], biases[:-1]):
        z = z.dot(w) + b
        z = sigmoid(z)
    logits = z.dot(weights[-1]) + biases[-1]
    preds = softmax(logits)
    return preds


def loss(y_true, y_pred):
    pass


def f(pos):
    """ Overhead function to train the network using the swarm"""
    ws, bs = unpack(pos, net_structure)
    y_pred = forward(X_train, ws, bs)
    return log_loss(y_train, y_pred)


if __name__ == '__main__':
    data = load_iris()
    X = data.data
    X_train = X[:125, :]
    X_test = X[125:, :]
    y = data.target
    y_train = y[:125]
    y_test = y[125:]
    net_structure = [X.shape[1], 16, 8, np.unique(y).size]

    for i in range(10, 21):
        swarm = GBestPSO(1000, np.prod(net_structure))
        c, p = swarm.minimize(f, iters=i)
        ws, bs = unpack(p, net_structure)
        preds = forward(X_test, ws, bs)
        y_pred = np.array(list(map(lambda x: np.argmax(x), preds)))
        print(f'i = {i}, acc = {accuracy_score(y_test, y_pred)}')
