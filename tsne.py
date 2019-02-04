#
#  tsne.py
#
# Implementation of t-SNE in Python. The implementation was tested on Python
# 2.7.10, and it requires a working installation of NumPy. The implementation
# comes with an example on the MNIST dataset. In order to plot the
# results of this example, a working installation of matplotlib is required.
#
# The example can be run by executing: `ipython tsne.py`
#
#
#  Created by Laurens van der Maaten on 20-12-08.
#  Copyright (c) 2008 Tilburg University. All rights reserved.

import numpy as np
import pylab


def Hbeta(D=np.array([]), beta=1.0):
    """
        Compute the perplexity and the P-row for a specific value of the
        precision of a Gaussian distribution.
    """

    # Compute P-row and corresponding perplexity
    P = np.exp(-D.copy() * beta)
    sumP = sum(P)
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P = P / sumP
    return H, P


def x2p(X=np.array([]), tol=1e-5, perplexity=30.0):
    """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.
    """

    # Initialize some variables
    print("Computing pairwise distances...")
    (n, d) = X.shape
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        if i % 500 == 0:
            print("Computing P-values for point %d of %d..." % (i, n))

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -np.inf
        betamax = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
        (H, thisP) = Hbeta(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP

    # Return final P-matrix
    print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
    return P


def pca(X=np.array([]), no_dims=50):
    """
        Runs PCA on the NxD array X in order to reduce its dimensionality to
        no_dims dimensions.
    """

    print("Preprocessing the data using PCA...")
    (n, d) = X.shape
    X = X - np.tile(np.mean(X, 0), (n, 1))
    (l, M) = np.linalg.eig(np.dot(X.T, X))
    Y = np.dot(X, M[:, 0:no_dims])
    return Y


def tsne(X=np.array([]), no_dims=2, initial_dims=50, perplexity=30.0):
    """
        Runs t-SNE on the dataset in the NxD array X to reduce its
        dimensionality to no_dims dimensions. The syntaxis of the function is
        `Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array.
    """

    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array X should have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    # Initialize variables
    X = pca(X, initial_dims).real
    (n, d) = X.shape
    max_iter = 1000
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    Y = np.random.randn(n, no_dims)
    dY = np.zeros((n, no_dims))
    iY = np.zeros((n, no_dims))
    gains = np.ones((n, no_dims))

    # Compute P-values
    P = x2p(X, 1e-5, perplexity)
    P = P + np.transpose(P)
    P = P / np.sum(P)
    P = P * 4.									# early exaggeration
    P = np.maximum(P, 1e-12)

    # Run iterations
    for iter in range(max_iter):

        # Compute pairwise affinities
        sum_Y = np.sum(np.square(Y), 1)
        num = -2. * np.dot(Y, Y.T)
        num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
        num[range(n), range(n)] = 0.
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)

        # Compute gradient
        PQ = P - Q
        for i in range(n):
            dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)

        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
                (gains * 0.8) * ((dY > 0.) == (iY > 0.))
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - np.tile(np.mean(Y, 0), (n, 1))

        # Compute current value of cost function
        if (iter + 1) % 10 == 0:
            C = np.sum(P * np.log(P / Q))
            print("Iteration %d: error is %f" % (iter + 1, C))

        # Stop lying about P-values
        if iter == 100:
            P = P / 4.

    # Return solution
    return Y



from kmedoids import k_medoids_selection
from acquisition import clustered_acquisition
from opts import parse_opts
import random
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
import matlab
import matlab.engine

def plot_heatmap(ax, Y_s, labels_s, x_min, x_max, y_min, y_max):
    clf = SVC(gamma=2, C=1, probability=True)
    y = labels_s
    X = Y_s
    h = .02
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # just plot the dataset first
    cm =  plt.cm.coolwarm # plt.cm.seismic #plt.cm.RdBu
    # cm_bright = ListedColormap(['red', 'blue', 'green', 'yellow', 'magenta'])

    # ax.set_title("Input data")
    clf.fit(X, y)
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    # Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)
    # Plot the training points
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())

if __name__ == "__main__":
    args = parse_opts()
    print("Run Y = tsne.tsne(X, no_dims, perplexity) to perform t-SNE on your dataset.")
    print("Running example on 9537 UCF101 ResNet Features...")
    labels =  np.load('ucf101_labels.npy')
    X = np.load('ucf101_resnet18_features.npy')
    # Y = tsne(X, 2, 50, 20.0)
    # np.save('ucf101_resnet18_tsne.npy', Y)
    Y = np.load('ucf101_resnet18_tsne.npy')

    cm_bright = plt.cm.bwr #plt.cm.seismic #plt.cm.RdBu # ListedColormap(['#FF0000', '#0000FF'])
    # datasets = [random.sample(set(labels), 2) for _ in range(10)]
    # datasets = [[8, 79], [80, 100], [7, 8], [0, 39], [5, 65], [7, 18], [8, 66], [24, 54], [25, 32], [25, 63],
    #             [39, 61], [40, 18], [27, 64], [40, 24], [50, 62], [52, 0], [57, 75], [67, 37], [74, 46], [80, 75],
    #             [83, 40], [83, 48], [91, 87], [100, 98]]
    # datasets = [[0, 39], [40, 18], [7, 18], [24, 54], [25, 32], [40,24], [50,62], [80, 100], [91, 87], [100, 98]]
    #
    #datasets = [[50, 62]]
    datasets = [[0, 39]]
    samp_per_clust = [2, 5, 10]
    for subset_classes in datasets:
        print subset_classes
        X_s = []
        Y_s = []
        labels_s = []
        i = 0
        for c in subset_classes:
            ind_c = np.where(labels == c)[0]
            X_s.extend(X[ind_c, :])
            Y_s.extend(Y[ind_c, :])
            labels_s.extend(i * np.ones(len(ind_c), dtype=np.int64))
            i += 1
        print set(labels_s)

        X_s = np.array(X_s)
        Y_s = np.array(Y_s)

        Y_s = StandardScaler().fit_transform(Y_s)


        x_min, x_max = Y_s[:, 0].min() - .5, Y_s[:, 0].max() + .5
        y_min, y_max = Y_s[:, 1].min() - .5, Y_s[:, 1].max() + .5

        labels_s = np.array(labels_s)

        fig = plt.figure(figsize=(2, 2))
        plt.subplots_adjust(left=0.02, bottom=0.02, right=0.98, top=0.98,
                                    wspace=0.03, hspace=0.03)
        ax = plt.subplot(1, 1, 1)
        plot_heatmap(ax, Y_s, labels_s, x_min, x_max, y_min, y_max)
        ax.scatter(Y_s[:, 0], Y_s[:, 1], c=labels_s, edgecolors='k', cmap=cm_bright)
        plt.savefig('data/results/tsne/' + 'original' + '.eps')
        #if s == 0:
        #    ax.set_title('Original')

        fig = plt.figure(figsize=(4, 4))
        plt.rc('text', usetex=True)
        plt.subplots_adjust(left=0.06, bottom=0.02, right=0.98, top=0.94,
                                    wspace=0.03, hspace=0.03)

        for s in range(len(samp_per_clust)):
            args.n_pool_clust = samp_per_clust[s]

            #if s == 0:
            #    fig = plt.figure(figsize=(5, 1.5))
            #    plt.subplots_adjust(left=0.02, bottom=0.02, right=0.98, top=0.85,
             #                       wspace=0.03, hspace=None)
            #else:
            #    fig = plt.figure(figsize=(5, 1.3))
            #    plt.subplots_adjust(left=0.02, bottom=0.02, right=0.98, top=0.98,
            #                        wspace=0.03, hspace=None)

            # pylab.scatter(Y_s[:, 0], Y_s[:, 1], 20, labels_s)

            clust = labels_s
            # clust = np.zeros(len(X))
            args.n_clust = len(set(clust))
            args.alpha = 1

            args.optimality = 'IPM'
            ipm = clustered_acquisition([], [], X_s, clust, np.random.rand(len(X_s)), args, args.n_clust * args.n_pool_clust)
            ax = plt.subplot(3, 3, s + 1 + (len(samp_per_clust)) * 2)
            plot_heatmap(ax, Y_s[ipm, :], labels_s[ipm], x_min, x_max, y_min, y_max)
            ax.scatter(Y_s[ipm, 0], Y_s[ipm, 1], c=labels_s[ipm], edgecolors='k', cmap=cm_bright)
            # ax.scatter(Y_s[ipm, 0], Y_s[ipm, 1], s=40, c='red', marker='v')
            if s == 0:
                ax.set_ylabel('IPM', fontsize=14)


            args.optimality = 'ds3'
            ds3 = clustered_acquisition([], [], X_s, clust, np.random.rand(len(X_s)), args, args.n_clust * args.n_pool_clust)
            ax = plt.subplot(3, 3, s + 1 + (len(samp_per_clust)) * 1)
            plot_heatmap(ax, Y_s[ds3, :], labels_s[ds3], x_min, x_max, y_min, y_max)
            ax.scatter(Y_s[ds3, 0], Y_s[ds3, 1], c=labels_s[ds3], edgecolors='k', cmap=cm_bright)
            if s == 0:
                ax.set_ylabel('DS3', fontsize=14)

            args.optimality = 'kmedoids'
            kmed = clustered_acquisition([], [], X_s, clust, np.random.rand(len(X_s)), args, args.n_clust * args.n_pool_clust)
            ax = plt.subplot(3, 3, s + 1)
            plot_heatmap(ax, Y_s[kmed, :], labels_s[kmed], x_min, x_max, y_min, y_max)
            ax.scatter(Y_s[kmed, 0], Y_s[kmed, 1], c=labels_s[kmed], edgecolors='k', cmap=cm_bright)
            ax.set_title(str(args.n_pool_clust) + ' Samples')
            if s == 0:
                ax.set_ylabel('K-medoids', fontsize=14)

            # ax.scatter(Y_s[:, 0], Y_s[:, 1], c=labels_s, edgecolors='k', cmap=cm_bright)
            # plt.figure()
            # plt.scatter(Y_s[:, 0], Y_s[:, 1], 20, labels_s)
            # pylab.scatter(Y_s[kmed, 0], Y_s[kmed, 1], s=40, c='red', marker='v')

            # pylab.figure()
            # pylab.scatter(Y_s[kmed, 0], Y_s[kmed, 1], 20, labels_s[kmed])
            # plt.savefig('data/results/tsne/' + str(subset_classes) + str(args.n_pool_clust) + 'samples'+  '.eps')
            plt.savefig('data/results/tsne/' + 'selections' + '.eps')

            # plt.show()
