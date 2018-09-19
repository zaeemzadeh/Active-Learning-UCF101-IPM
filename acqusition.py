import torch
from torch.autograd import Variable
import numpy as np
import sklearn.cluster as cl


def acqusition(pool_loader, train_loader, model, opts):
    if opts.score_func == 'random':
        score = np.random.rand(len(pool_loader.dataset))
    else:
        raise ValueError('Invalid score function for data selection!')

    if opts.alpha == 0:  # only score func (no eig optimality)
        pooled_idx = np.argsort(score)[-opts.n_pool:]
    else:
        pool_features = extract_features(pool_loader, model)
        train_features = extract_features(train_loader, model)
        pooled_idx = e_optimal_clustered_acquisition(train_features, pool_features, score, opts)
    return pooled_idx

def extract_features(data_loader, model):
    model.eval()
    features = []
    for i, (inputs, _) in enumerate(data_loader):
        with torch.no_grad():
            inputs = Variable(inputs)
        outputs = model(inputs).data
        # TODO: convert to cupy more efficiently
        #features.extend(cp.array(outputs.cpu().numpy()) )
        features.extend(outputs.cpu().numpy())
        if i % 100 == 0:
            print('Feature Extraction Batch: [{0}/{1}]'.format(i + 1, len(data_loader)))
    return features


def e_optimal_clustered_acquisition(f_train, f_pool, score, args):
    # clustering data in the feature space
    clust_pool, clust_train = feature_clust(f_pool, f_train, args.n_clust)
    pooled_idx = []
    # optimal selction in each cluster
    for c in range(args.n_clust):
        idx_pool_c  = np.where(clust_pool == c)[0]
        idx_train_c = np.where(clust_train == c)[0]

        f_pool_c = [f_pool[i][0] for i in idx_pool_c]
        f_train_c = [f_train[i][0] for i in idx_train_c]

        score_c = np.asarray([float(score[i]) for i in idx_pool_c])

        #n_pool_clust = np.minimum(n_pool, len(score_c))
        n_pool_clust = np.minimum(args.n_pool_clust, len(score_c))
        pooled_idx_c = e_optimal_acquisition(f_train_c, f_pool_c, score_c, n_pool_clust, args.alpha, type=args.optimality)

        pooled_idx.extend([int(idx_pool_c[i]) for i in pooled_idx_c])

    # kemoids on the selected samples
    # pooled_features = [f_pool[i][0] for i in pooled_idx]
    # dist_mat = pairwise_distances(pooled_features, metric='euclidean')
    # med_idx, _ = kmedoids.kMedoids(dist_mat, n_pool)
    #
    # return [int(pooled_idx[i]) for i in med_idx]

    # uncertainty selection on the selected samples
    pooled_score = [float(score[i]) for i in pooled_idx]
    sorted_idx = np.argsort(pooled_score)
    sorted_idx = sorted_idx[-args.n_pool:]
    return [int(pooled_idx[i]) for i in sorted_idx]


def feature_clust(f_pool, f_train, n_clust):
    data_f_pool = f_pool
    data_f_train = f_train

    data_f_pool.extend(data_f_train)

    spectral = cl.SpectralClustering(n_clusters=n_clust, eigen_solver='arpack', affinity="nearest_neighbors")
    spectral.fit(data_f_pool)
    labels = spectral.labels_

    # clusters = cl.k_means(data_f_pool, 10)  #Kmeans Clustering
    # labels = clusters[1]

    clust_pool = labels[0:-len(f_train)]
    clust_train = labels[-len(f_train):]

    return clust_pool, clust_train

def e_optimal_acquisition(train, pool, score, n_pool, alpha, type):
    #pooled_idx = [int(cp.argmax(score))]
    pooled_idx = []
    while len(pooled_idx) < n_pool:
        if type == 'IPM':
            new_idx = IPM_add_sample(train, pool, pooled_idx)
        elif type == 'MP':
            new_idx = MP_add_sample(train, pool, pooled_idx)
        else:
            new_idx = e_optimal_add_sample(train, pool, score, pooled_idx, alpha, type)

        pooled_idx.append(int(new_idx))
    return pooled_idx


def e_optimal_add_sample(train, pool, score, pooled_idx, alpha, type):
    candidate_samples = range(0, len(pool))         # all samples
    # candidate_samples = cp.argsort(score)[-100:]     # best samples based on score
    A_train = [np.ravel(t) for t in train]
    #if len(A_train) > 0:
    #    u, s, v = np.linalg.svd(A_train)
    #    A_train = [np.ravel(v[i]) for i in range(np.minimum(len(A_train), 5))]


    # calculating eig-based score
    eig_score = np.zeros(len(pool))
    for m in candidate_samples:

        if m in pooled_idx:
            continue

        set_idx = [int(idx) for idx in pooled_idx]
        set_idx.append(int(m))

        A = [np.ravel(pool[i]) for i in set_idx]
        A.extend(A_train)
        A = [a/np.linalg.norm(a) for a in A]            # normalization
        A = np.asarray(A)

        eigs = np.linalg.eigvalsh(np.matmul(A, A.transpose()) + 0.5*np.eye(len(A)))

        if type == 'd_optimal':
            eig_score[m] = np.prod(eigs)                  # determinant
        elif type == 'e_optimal':
            eig_score[m] = np.min(eigs)                     # minimum eigen value      # determinant
        elif type == 'a_optimal':
            eig_score[m] = np.sum(eigs)                     # trace      # determinant
        elif type == 'inv_cond':
            eig_score[m] = np.min(eigs)/np.max(eigs)        # inverse condition number
        else:
            raise ValueError('Invalid optimality for data selection!')

    # finding the best sample
    eig_score /= np.ndarray.max(eig_score)
    objective = (1-alpha)*score + alpha*eig_score
    # objective = eig_score
    sorted_idx = np.argsort(objective)

    # avoiding duplicates
    i = 1
    while sorted_idx[-i] in pooled_idx:
        i += 1
    #print eig_score[sorted_idx[-i]]
    return sorted_idx[-i]



