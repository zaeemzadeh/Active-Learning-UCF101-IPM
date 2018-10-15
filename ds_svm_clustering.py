import numpy as np
from numpy.linalg import norm
from scipy.linalg import sqrtm
import random

from sklearn.metrics import pairwise_distances, adjusted_mutual_info_score
from sklearn.datasets import make_classification
from sklearn.cluster import SpectralClustering
import sklearn.svm as svm
import matplotlib.pyplot as plt
import multiprocessing


def inv_conv(X):
    import torch
    X = np.array(X)
    Xt = torch.from_numpy(X).type(torch.double)
    Xt -= torch.mean(Xt, dim=0, keepdim=True)

    fact = 1.0 / (Xt.size(0) - 1)
    Cov = fact * Xt.t().matmul(Xt).squeeze()
    Cov_inv = torch.inverse(Cov)
    return Cov_inv.data.numpy().astype(np.float_)


def compute_kernel(X, Y=None, metric='euclidean'):
    X = np.array(X)
    D = pairwise_distances(X, Y, metric=metric)
    gamma = 1.0
    # gamma = 1.0 / (X.std() * X.shape[1])
    # gamma = 1.0 / np.median(D)
    # D *= gamma
    # D = D**2
    S = np.exp(-gamma * D**2 / 2.0)
    return S


def dominant_set(A, x=None, epsilon=1.0e-4):
    """Compute the dominant set of the similarity matrix A with the
    replicator dynamics optimization approach. Convergence is reached
    when x changes less than epsilon.

    See: 'Dominant Sets and Pairwise Clustering', by Massimiliano
    Pavan and Marcello Pelillo, PAMI 2007.
    """
    if x is None:
        x = np.ones(A.shape[0]) / float(A.shape[0])

    distance = epsilon * 2
    while distance > epsilon:
        x_old = x.copy()
        # x = x * np.dot(A, x) # this works only for dense A
        x = x * A.dot(x)  # this works both for dense and sparse A
        x = x / x.sum()
        distance = norm(x - x_old)
        # print x.size, distance

    return x


def svm_ovr(args):
    [X, labels, non_dominant_set_idx, dominant_set_idx, l, eta, metric] = args
    if l == -1:
        return []

    if len(non_dominant_set_idx) == 0:
        return []

    n_select = eta if len(non_dominant_set_idx) >= eta else len(non_dominant_set_idx)

    ovr_classes = [1 if labels[dominant_set_idx[i]] == l else -1 for i in range(len(dominant_set_idx))]

    clf = svm.SVC(kernel='precomputed', tol=1e-5)

    X_d = [X[i, :] for i in dominant_set_idx]
    S = compute_kernel(X_d, metric=metric)

    clf.fit(S, ovr_classes)

    X_nd = [X[i, :] for i in non_dominant_set_idx]
    S = compute_kernel(X_nd, X_d, metric=metric)
    scores = clf.decision_function(S)

    new_idx_l = np.argsort(scores)[-n_select:]
    return new_idx_l, l


def ds_svm_clustering(X, n_clust=2, eta=2, ds_ratio=0.25, plot=False, metric='euclidean'):
    """Dominant set + SVM Clustering:
    Alg 1 in Unsupervised Action Discovery and Localization in Videos
    http://crcv.ucf.edu/papers/iccv17/Soomro_ICCV17.pdf
    Written by: Alireza Zaeemzadeh
    """
    X = np.array(X)
    if plot:
        X_orig = X.copy()
        plt.figure()
        plt.subplot(2, 2, 1)
        for yi in np.unique(y):
            plt.plot(X_orig[y==yi,0], X_orig[y==yi,1], 'o')

        plt.title('Dataset')

    # normalizing data
    VI = inv_conv(X)
    X = np.matmul(sqrtm(VI), (X - np.mean(X, axis=0, keepdims=True)).transpose()).transpose()

    S = compute_kernel(X, metric=metric)

    print 'SpectralClustering'
    spectral = SpectralClustering(n_clusters=n_clust, eigen_solver='arpack', affinity='precomputed')
    spectral.fit(S)
    labels = spectral.labels_

    if plot:
        plt.subplot(2, 2, 2)
        for l in np.unique(labels):
            plt.plot(X_orig[labels == l, 0], X_orig[labels == l, 1], 'o')

        plt.title('Spectral Clustering')

    # finding dominant sets
    print 'finding dominant sets'
    for l in np.unique(labels):
        idx_l = np.where(labels == l)
        if len(idx_l[0]) < 4:
            print l, len(idx_l[0])
            continue

        X_l = X[labels == l, :]
        X_l = np.matmul(sqrtm(inv_conv(X_l)), (X_l - np.mean(X_l, axis=0, keepdims=True)).transpose()).transpose()
        S_l = compute_kernel(X_l, metric=metric)

        x = dominant_set(S_l, epsilon=2e-3)

        cutoff = np.percentile(x[x > 0], 100. * ( 1 - ds_ratio))
        dom_idx = x > cutoff

        for i in idx_l[0][~dom_idx]:
            labels[i] = -1
    if plot:
        plt.subplot(2, 2, 3)
        for yi in np.unique(labels):
            plt.plot(X_orig[labels == yi, 0], X_orig[labels == yi, 1], 'o')

        plt.title('Dominant sets')

    dominant_set_idx = np.where(labels != -1)[0].tolist()
    non_dominant_set_idx = np.where(labels == -1)[0].tolist()
    print 'SVM loop'
    n_proc = multiprocessing.cpu_count() - 2
    pool = multiprocessing.Pool(processes=n_proc)
    while len(non_dominant_set_idx) > 0:
        print len(non_dominant_set_idx), len(dominant_set_idx), len(non_dominant_set_idx) + len(dominant_set_idx)
        chunks = [[X, labels, non_dominant_set_idx, dominant_set_idx, l, eta, metric] for l in
                  np.unique(labels)]
        new_idx = pool.map(svm_ovr, chunks)

        remove_idx = []

        for worker in random.sample(range(len(new_idx)), k=len(new_idx)):
            # print new_idx[worker]
            if len(new_idx[worker]) == 0:
                continue

            new_idx_w = new_idx[worker][0]
            label_w = new_idx[worker][1]
            for i in new_idx_w:
                dominant_set_idx.append(non_dominant_set_idx[i])
                labels[non_dominant_set_idx[i]] = label_w

            remove_idx.extend(new_idx_w)

        dominant_set_idx = list(np.unique(dominant_set_idx))
        remove_idx  = list(np.unique(remove_idx))
        for idx in sorted(remove_idx, reverse=True):
            del non_dominant_set_idx[idx]

        # remove_idx = []
        # for l in np.unique(labels):
        #     new_idx_l = svm_ovr([X, labels, non_dominant_set_idx, dominant_set_idx, l, eta, metric])
        #     if len(new_idx_l) == 0:
        #         continue
        #     remove_idx.extend(new_idx_l[0])
        #
        #     for i in new_idx_l[0]:
        #         dominant_set_idx.append(non_dominant_set_idx[i])
        #         labels[non_dominant_set_idx[i]] = l
        #
        #     for idx in sorted(remove_idx, reverse=True):
        #         # print idx
        #         del non_dominant_set_idx[idx]
        #     remove_idx = []

    pool.terminate()
    if plot:
        plt.subplot(2, 2, 4)
        for yi in np.unique(labels):
            plt.plot(X_orig[labels == yi, 0], X_orig[labels == yi, 1], 'o')

        plt.title('Dominant set + SVM Clustering')

    return labels


if __name__ == '__main__':
    np.random.seed(6)

    nclust =3
    N = 1000    # number of samples
    d = 2    # dimension of samples (number of features)
    plot = True
    weights = np.ones(nclust)
    weights /= sum(weights)

    eta = 4

    X, y = make_classification(weights=weights.tolist(), n_classes=nclust, n_samples=N, n_features=d,
                               n_redundant=0, class_sep=1, n_clusters_per_class=1, n_informative=d)

    dist_metric = 'euclidean'  #cosine, euclidean, l1, l2, manhattan, mahalanobis
    labels = ds_svm_clustering(X, eta=eta, ds_ratio=0.25, n_clust=nclust, plot=plot, metric=dist_metric)
    print 'Adjusted Mutual Information Score: ', adjusted_mutual_info_score(y, labels)
    plt.show()