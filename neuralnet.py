import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import train_test_split
from pso import GBestPSO


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    ex = np.exp(x)
    emx = np.exp(-x)
    return (ex - emx) / (ex + emx)


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
        z = tanh(z)
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
    data = load_digits()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    net_structure = [X.shape[1], 32, np.unique(y).size]
    dimensions = np.prod(net_structure)
    bounds = (np.full((dimensions, 1), -1), np.full((dimensions, 1), 1))
    for i in range(50, 101, 10):
        swarm = GBestPSO(200, dimensions, bounds=bounds, verbose=True)
        c, p = swarm.minimize(f, iters=i)
        ws, bs = unpack(p, net_structure)
        preds = forward(X_test, ws, bs)
        y_pred = np.array(list(map(lambda x: np.argmax(x), preds)))
        print(f'i = {i}, acc = {accuracy_score(y_test, y_pred)}')
