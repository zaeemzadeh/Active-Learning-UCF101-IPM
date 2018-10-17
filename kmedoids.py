from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt


def k_medoids_selection(data, n_medoids):
    data = np.array(data)
    D = pairwise_distances(data, metric='euclidean')


    # split into clusters
    medoids, labels = kMedoids(D, n_medoids)
    medoids.astype(np.int64)

    clusters = np.zeros(len(data))
    for label in labels:
        for point_idx in labels[label]:
            clusters[point_idx] = label

    return medoids, clusters


def kMedoids(D, k, tmax=100):
    # determine dimensions of distance matrix D
    m, n = D.shape

    if k > n:
        raise Exception('too many medoids')

    # find a set of valid initial cluster medoid indices since we
    # can't seed different clusters with two points at the same location
    valid_medoid_inds = set(range(n))
    invalid_medoid_inds = set([])
    rs,cs = np.where(D==0)
    # the rows, cols must be shuffled because we will keep the first duplicate below
    index_shuf = range(len(rs))
    np.random.shuffle(index_shuf)
    rs = rs[index_shuf]
    cs = cs[index_shuf]
    for r,c in zip(rs,cs):
        # if there are two points with a distance of 0...
        # keep the first one for cluster init
        if r < c and r not in invalid_medoid_inds:
            invalid_medoid_inds.add(c)
    valid_medoid_inds = list(valid_medoid_inds - invalid_medoid_inds)

    if k > len(valid_medoid_inds):
        raise Exception('too many medoids (after removing {} duplicate points)'.format(
            len(invalid_medoid_inds)))

    # randomly initialize an array of k medoid indices
    M = np.array(valid_medoid_inds)
    np.random.shuffle(M)
    M = np.sort(M[:k])

    # create a copy of the array of medoid indices
    Mnew = np.copy(M)

    # initialize a dictionary to represent clusters
    C = {}
    for t in xrange(tmax):
        # determine clusters, i. e. arrays of data indices
        J = np.argmin(D[:,M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J==kappa)[0]
        # update cluster medoids
        for kappa in range(k):
            J = np.mean(D[np.ix_(C[kappa],C[kappa])],axis=1)
            j = np.argmin(J)
            Mnew[kappa] = C[kappa][j]
        np.sort(Mnew)
        # check for convergence
        if np.array_equal(M, Mnew):
            break
        M = np.copy(Mnew)
    else:
        # final update of cluster memberships
        J = np.argmin(D[:,M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J==kappa)[0]

    # return results
    return M, C


if __name__ == '__main__':

    np.random.seed(6)

    # 3 points in dataset

    nclust = 3
    N = 9537   # number of samples
    d = 512  # dimension of samples (number of features)
    plot = False
    weights = np.ones(nclust)
    weights /= sum(weights)

    data, y = make_classification(weights=weights.tolist(), n_classes=nclust, n_samples=N, n_features=d,
                               n_redundant=0, class_sep=1, n_clusters_per_class=1, n_informative=d)

    # distance matrix
    D = pairwise_distances(data, metric='euclidean')

    # split into clusters
    M, C = kMedoids(D, nclust)
    clusters = np.zeros(len(data), dtype=np.int64)
    for label in C:
        for point_idx in C[label]:
            clusters[point_idx] = int(label)

    if plot:
        for label in C:
            plt.plot(data[C[label], 0], data[C[label], 1], 'o')

        plt.title('kmedoids')
        plt.show()

    # print('medoids:')
    # for point_idx in M:
    #     print(data[point_idx])

    # print('')
    # print('clustering result:')
    # for label in C:
    #     for point_idx in C[label]:
    #         print('label {0}: {1}'.format(label, data[point_idx]))