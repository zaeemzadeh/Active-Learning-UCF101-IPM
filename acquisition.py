import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import irlb


def acquisition(pool_loader, train_loader, model, opts):
    # creating loaders without shuffles
    pool_loader_noshuffle = torch.utils.data.DataLoader(
        pool_loader.dataset,
        batch_size=opts.batch_size,
        shuffle=False,
        num_workers=opts.n_threads,
        pin_memory=True)

    train_loader_noshuffle = torch.utils.data.DataLoader(
        train_loader.dataset,
        batch_size=opts.batch_size,
        shuffle=False,
        num_workers=opts.n_threads,
        pin_memory=True)

    # setting the number of the samples to be pooled
    if len(train_loader.dataset.indices) == 0: # initial acquisition
        print 'initial selection: n_pool = ', opts.init_train_size
        n_pool = opts.init_train_size
    else:
        print 'n_pool = ', opts.n_pool
        n_pool = opts.n_pool

    print 'extracting features of the training dataset. '
    train_features, train_labels = extract_features(train_loader_noshuffle, model)
    print 'extracting features of the pooling dataset.'
    pool_features, pool_labels = extract_features(pool_loader_noshuffle, model)

    print 'selecting samples.'
    pooled_idx = ipm(train_features, pool_features, n_pool)

    pooled_idx_set = set([pool_loader_noshuffle.dataset.indices[i] for i in pooled_idx])

    train_loader.dataset.indices = list(set(train_loader.dataset.indices) | pooled_idx_set)
    pool_loader.dataset.indices = list(set(pool_loader.dataset.indices) - pooled_idx_set)

    return


def extract_features(data_loader, model, label_only=False):
    if not label_only:
        feature_extractor = nn.Sequential(*list(model.module.children())[:-2])
        feature_extractor = feature_extractor.cuda()
        feature_extractor = nn.DataParallel(feature_extractor, device_ids=None)
        feature_extractor.eval()

    features = []
    labels = []
    for i, (inputs, l) in enumerate(data_loader):
        labels.extend(l.data.cpu().numpy())

        if not label_only:
            with torch.no_grad():
                inputs = Variable(inputs)
                batch_features = feature_extractor(inputs).data.view(inputs.size(0), -1)

            # TODO: convert to numpy more efficiently
            features.extend(batch_features.cpu().numpy())

        if i % 100 == 0:
            print('[{0}/{1}]'.format(i + 1, len(data_loader)))

    return features, labels


def ipm(train, pool, n_pool):
    pooled_idx = []
    while len(pooled_idx) < n_pool:
        new_idx = ipm_add_sample(train, pool, pooled_idx)
        pooled_idx.append(int(new_idx))
    return pooled_idx


def ipm_add_sample(train, pool, pooled_idx):
    candidate_samples = range(0, len(pool))         # all samples
    set_idx = [int(idx) for idx in pooled_idx]

    A_train = [np.ravel(t) for t in train]
    A_train.extend([np.ravel(pool[i]) for i in set_idx])
    A_s_mat = np.array(A_train).transpose()
    if len(A_s_mat.shape) == 1:
        A_s_mat = A_s_mat.reshape((-1, 1))

    A_pool = [np.ravel(t) for t in pool]
    A_mat = np.array(A_pool).transpose()
    if len(A_mat.shape) == 1:
        A_mat = A_mat.reshape((-1, 1))

    if len(A_s_mat) == 0:
        A_proj = A_mat
    else:
        Proj = np.matmul(A_s_mat, np.linalg.pinv(A_s_mat))
        A_proj = A_mat - np.matmul(Proj, A_mat)

    # u, _, _ = np.linalg.svd(A_proj, full_matrices=False)
    # first_eig_vec = u[:, 0]

    first_eig_vec = irlb.irlb(A_proj, 2)[0][:, 0]

    # calculating score
    correlation = np.zeros(len(pool))
    for m in candidate_samples:

        if m in pooled_idx:
            correlation[m] = 0
            continue

        correlation[m] = np.abs(np.inner(A_mat[:, m], first_eig_vec))
        # correlation[m] = np.inner(A_mat[:, m], first_eig_vec)
        correlation[m] /= np.linalg.norm(np.squeeze(A_mat[:, m]))

    # finding the best sample
    objective = correlation
    sorted_idx = np.argsort(objective)
    sorted_idx = np.flipud(sorted_idx)   # sort in decsending order

    # avoiding duplicates
    i = 0
    while sorted_idx[i] in pooled_idx:
        i += 1

    return sorted_idx[i]