#!/usr/bin/env python

# Deep Learning Homework 1

import argparse

import numpy as np
import matplotlib.pyplot as plt

import utils


class LinearModel(object):
    def __init__(self, n_classes, n_features, **kwargs):
        self.W = np.zeros((n_classes, n_features))

    def update_weight(self, x_i, y_i, **kwargs):
        raise NotImplementedError

    def train_epoch(self, X, y, **kwargs):
        for x_i, y_i in zip(X, y):
            self.update_weight(x_i, y_i, **kwargs)

    def predict(self, X):
        """X (n_examples x n_features)"""
        scores = np.dot(self.W, X.T)  # (n_classes x n_examples)
        predicted_labels = scores.argmax(axis=0)  # (n_examples)
        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features):
        y (n_examples): gold labels
        """
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible


class Perceptron(LinearModel):
    def update_weight(self, x_i, y_i, **kwargs):
        """
        x_i (n_features): a single training example
        y_i (scalar): the gold label for that example
        other arguments are ignored
        """
        # Q1.1a
        z_i = self.predict(x_i)
        if z_i != y_i:
            self.W[y_i] += x_i
            self.W[z_i] -= x_i


class LogisticRegression(LinearModel):
    def update_weight(self, x_i, y_i, learning_rate=0.001):
        """
        x_i (n_features): a single training example
        y_i: the gold label for that example
        learning_rate (float): keep it at the default value for your plots
        """
        def gradient(W, x, y):
             # Q1.1b
            n_classes = len(W[:,0])
            n_features = len(W[0,:])
            e_y = np.zeros((n_classes,1))
            e_y[y] = 1
            prob_vec = np.exp(np.matmul(W ,x))
            Z_x = np.sum(prob_vec)
            prob_vec = np.divide(prob_vec,Z_x)
            grad = np.matmul((prob_vec - e_y.T).T,np.reshape(x,(1,n_features)))

            return grad
        
        W_k = self.W
        grad = gradient(W_k, x_i, y_i)
        W_k1 = W_k - learning_rate * grad

        self.W = W_k1


class MLP(object):
    # Q3.2b. This MLP skeleton code allows the MLP to be used in place of the
    # linear models with no changes to the training loop or evaluation code
    # in main().
    def __init__(self, n_classes, n_features, hidden_size):
        # Initialize an MLP with a single hidden layer.
        self.W1 = np.random.normal(0.1, 0.1, (hidden_size, n_features))
        self.b1 = np.zeros((hidden_size,1))

        self.W2 = np.random.normal(0.1, 0.1, (n_classes, hidden_size))
        self.b2 = np.zeros((n_classes,1))

        self.n_classes = n_classes


    def forwardPropagation(self, X):
        #DIMENSIONS
        # X: (n_data, n_feat)
        # z1: (n_data, n_hidden)
        # h1: (n_data. n_hidden)
        # z2: (n_data, n_class)
        # Z (n_data,)
        # score (n_data, n_class)

        z1 = (np.dot(self.W1, X.T)+self.b1).T
        h1 = np.maximum(0, z1) #RELU

        z2 = (np.dot(self.W2, h1.T)+self.b2).T

        exp = np.exp(z2.T - np.max(z2,axis=1).T).T
        Z = np.sum(exp, axis=1)

        scores = (exp.T / Z.T).T

        return z1, h1, z2, scores
        
    def predict(self, X):

        # Compute the forward pass of the network. At prediction time, there is
        # no need to save the values of hidden nodes, whereas this is required
        # at training time.

        z1, h1, z2, scores = self.forwardPropagation(X)

        return scores.argmax(axis=1)

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        # Identical to LinearModel.evaluate()
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible

    def train_epoch(self, X, y, learning_rate=0.001):
        """
        Dont forget to return the loss of the epoch.
        """

        def crossEntropyLoss(score, oneHot):
            return -np.sum(np.multiply(np.log(score), oneHot))
               
        oneHot = np.zeros((y.size, self.n_classes))
        oneHot[np.arange(y.size), y] = 1
        
        for i in range(len(y)):
            x = X[i].reshape(1, len(X[i]))
            z1, h1, z2, scores = self.forwardPropagation(x)

            dz2 = (scores - oneHot[i])
            dW2 = np.dot(dz2.T, h1)
            db2 = dz2.T

            dh1 = np.dot(self.W2.T, dz2.T).T
            dz1 = np.multiply(dh1, np.heaviside(z1, 0))
            dW1 = np.dot(dz1.T, x)
            db1 = dz1.T

            self.W2 -= learning_rate * dW2
            self.b2 -= learning_rate * db2
            self.W1 -= learning_rate * dW1
            self.b1 -= learning_rate * db1

        return crossEntropyLoss(self.forwardPropagation(X)[3], oneHot)


def plot(epochs, train_accs, val_accs):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(epochs, train_accs, label='train')
    plt.plot(epochs, val_accs, label='validation')
    plt.legend()
    plt.show()

def plot_loss(epochs, loss):
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(epochs, loss, label='train')
    plt.legend()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['perceptron', 'logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-hidden_size', type=int, default=200,
                        help="""Number of units in hidden layers (needed only
                        for MLP, not perceptron or logistic regression)""")
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Learning rate for parameter updates (needed for
                        logistic regression and MLP, but not perceptron)""")
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    add_bias = opt.model != "mlp"
    data = utils.load_oct_data(bias=add_bias)
    train_X, train_y = data["train"]
    dev_X, dev_y = data["dev"]
    test_X, test_y = data["test"]
    n_classes = np.unique(train_y).size
    n_feats = train_X.shape[1]

    # initialize the model
    if opt.model == 'perceptron':
        model = Perceptron(n_classes, n_feats)
    elif opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats)
    else:
        model = MLP(n_classes, n_feats, opt.hidden_size)
    epochs = np.arange(1, opt.epochs + 1)
    train_loss = []
    valid_accs = []
    train_accs = []
    
    for i in epochs:
        print('Training epoch {}'.format(i))
        train_order = np.random.permutation(train_X.shape[0])
        train_X = train_X[train_order]
        train_y = train_y[train_order]
        if opt.model == 'mlp':
            loss = model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate
            )
        else:
            model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate
            )
        
        train_accs.append(model.evaluate(train_X, train_y))
        valid_accs.append(model.evaluate(dev_X, dev_y))
        if opt.model == 'mlp':
            print('loss: {:.4f} | train acc: {:.4f} | val acc: {:.4f}'.format(
                loss, train_accs[-1], valid_accs[-1],
            ))
            train_loss.append(loss)
        else:
            print('train acc: {:.4f} | val acc: {:.4f}'.format(
                 train_accs[-1], valid_accs[-1],
            ))
    print('Final test acc: {:.4f}'.format(
        model.evaluate(test_X, test_y)
        ))

    # plot
    plot(epochs, train_accs, valid_accs)
    if opt.model == 'mlp':
        plot_loss(epochs, train_loss)


if __name__ == '__main__':
    main()
