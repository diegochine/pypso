import numpy as np
import matplotlib.pyplot as plt
from pso.gbest import GBestPSO


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


def log_loss(y_true, y_pred):
    # compute loss per each class
    per_class_loss = np.concatenate([-np.log(y_pred[y_true == c, c]) for c in np.unique(y_true)])
    loss = np.mean(per_class_loss)
    return loss


def f(pos):
    """ Overhead function to train the network using the swarm"""
    ws, bs = unpack(pos, net_structure)
    y_pred = forward(X_train, ws, bs)
    y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)
    y_pred = y_pred / y_pred.sum(axis=0)
    return log_loss(y_train, y_pred)


if __name__ == '__main__':
    X = np.load('data/digits_data.npy')
    y = np.load('data/digits_labels.npy')
    X = X / np.max(X)
    X_train = X[:1300, :]
    X_test = X[1300:, :]
    y_train = y[:1300]
    y_test = y[1300:]
    net_structure = [X.shape[1], 32, np.unique(y).size]
    dimensions = int(np.prod(net_structure))
    bounds = (np.full((dimensions, 1), -0.5), np.full((dimensions, 1), 0.5))
    swarm = GBestPSO(150, dimensions, bounds=bounds, verbose=True)
    cost, solution = swarm.minimize(f, iters=50)
    ws, bs = unpack(solution, net_structure)
    preds = forward(X_test, ws, bs)
    y_pred = np.array(list(map(lambda x: np.argmax(x), preds)))
    accuracy = (y_test == y_pred).sum() / y_test.size
    print(f'Accuracy (testing set): {accuracy * 100:.2f}%')

    fig = plt.figure(figsize=(12, 7), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)
    choice = np.random.choice(X_test.shape[0], 3, replace=False)
    for ax_idx, img_idx in enumerate(choice):
        ax = fig.add_subplot(gs[0, ax_idx])
        ax.imshow(X_test[img_idx].reshape(8, 8), cmap='gray')
        ax.set_title(f'True label: {y_test[img_idx]}, predicted: {y_pred[img_idx]}')
    ax = fig.add_subplot(gs[1, :])
    ax.plot(swarm.cost_history)
    ax.set_title(f'Loss function, final accuracy: {accuracy * 100:.2f}%')
    ax.set_xlabel('#iterations')
    ax.set_ylabel('Loss')
    ax.grid()
    plt.show()
