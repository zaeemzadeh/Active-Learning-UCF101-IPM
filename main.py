import os
import sys
import json
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler

from opts import parse_opts
from model import generate_model
from mean import get_mean, get_std
from spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, CornerCrop, MultiScaleCornerCrop,
    MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor)
from temporal_transforms import LoopPadding, TemporalRandomCrop
from target_transforms import ClassLabel, VideoID
from target_transforms import Compose as TargetCompose
from dataset import get_training_set, get_validation_set, get_test_set
from utils import Logger
from train import train_epoch
from validation import val_epoch
from acqusition import acqusition
import test

if __name__ == '__main__':

    opt = parse_opts()
    if opt.root_path != '':
        opt.video_path = os.path.join(opt.root_path, opt.video_path)
        opt.annotation_path = os.path.join(opt.root_path, opt.annotation_path)
        opt.result_path = os.path.join(opt.root_path, opt.result_path)
        if opt.resume_path:
            opt.resume_path = os.path.join(opt.root_path, opt.resume_path)
        if opt.pretrain_path:
            opt.pretrain_path = os.path.join(opt.root_path, opt.pretrain_path)
    opt.scales = [opt.initial_scale]
    for i in range(1, opt.n_scales):
        opt.scales.append(opt.scales[-1] * opt.scale_step)
    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
    opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
    opt.std = get_std(opt.norm_value)
    print(opt)
    dir_path = 'resnet{}_shortcut{}_from_sctratch_model_bs{}'.format(opt.model_depth, opt.resnet_shortcut, opt.batch_size)
    opt.result_path = os.path.join(opt.result_path, dir_path)
    if not os.path.exists(opt.result_path):
        os.makedirs(opt.result_path)
    with open(os.path.join(opt.result_path, 'opts.json'), 'w') as opt_file:
        json.dump(vars(opt), opt_file)

    torch.manual_seed(opt.manual_seed)

    model, parameters = generate_model(opt)
    #print(model)
    criterion = nn.CrossEntropyLoss()
    if not opt.no_cuda:
        criterion = criterion.cuda()

    if opt.no_mean_norm and not opt.std_norm:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not opt.std_norm:
        norm_method = Normalize(opt.mean, [1, 1, 1])
    else:
        norm_method = Normalize(opt.mean, opt.std)

    if not opt.no_train:
        assert opt.train_crop in ['random', 'corner', 'center']
        if opt.train_crop == 'random':
            crop_method = MultiScaleRandomCrop(opt.scales, opt.sample_size)
        elif opt.train_crop == 'corner':
            crop_method = MultiScaleCornerCrop(opt.scales, opt.sample_size)
        elif opt.train_crop == 'center':
            crop_method = MultiScaleCornerCrop(
                opt.scales, opt.sample_size, crop_positions=['c'])
        spatial_transform = Compose([
            crop_method,
            RandomHorizontalFlip(),
            ToTensor(opt.norm_value), norm_method
        ])
        temporal_transform = TemporalRandomCrop(opt.sample_duration)
        target_transform = ClassLabel()
        labeled_data = get_training_set(opt, spatial_transform,
                                         temporal_transform, target_transform)
        """
        labeled_data should be an instance of a class which inherits torch.utils.data.Dataset class and overrides two methods: __len__ so that len(training_data) returns the size of the training dataset and __getitem__ to support the indexing such that training_data[i] can be used to get the ith sample.
        """
        # split data into train and pool datasets
        if opt.pretrain_path:
            # starting from empty training dataset, we will choose the training dataset using the prerained model
            training_idx_set = set()
            pool_idx_set = set(range(len(labeled_data)))
        else:
            # split data randomly
            training_idx_set = set(np.random.permutation(range(len(labeled_data)))[:opt.init_train_size])
            pool_idx_set = set(range(len(labeled_data))) - training_idx_set

        training_data = torch.utils.data.Subset(labeled_data, list(training_idx_set))
        pool_data = torch.utils.data.Subset(labeled_data, list(pool_idx_set))

        pool_loader = torch.utils.data.DataLoader(
            pool_data,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.n_threads,
            pin_memory=True)

        train_loader = torch.utils.data.DataLoader(
            training_data,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.n_threads,
            pin_memory=True)

        train_logger = Logger(
            os.path.join(opt.result_path, 'train.log'),
            ['epoch', 'loss', 'acc', 'lr'])
        train_batch_logger = Logger(
            os.path.join(opt.result_path, 'train_batch.log'),
            ['epoch', 'batch', 'iter', 'loss', 'acc', 'lr'])

        if opt.nesterov:
            dampening = 0
        else:
            dampening = opt.dampening
        optimizer = optim.SGD(
            parameters,
            lr=opt.learning_rate,
            momentum=opt.momentum,
            dampening=dampening,
            weight_decay=opt.weight_decay,
            nesterov=opt.nesterov)
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=opt.lr_patience)
    if not opt.no_val:
        spatial_transform = Compose([
            Scale(opt.sample_size),
            CenterCrop(opt.sample_size),
            ToTensor(opt.norm_value), norm_method
        ])
        temporal_transform = LoopPadding(opt.sample_duration)
        target_transform = ClassLabel()
        validation_data = get_validation_set(
            opt, spatial_transform, temporal_transform, target_transform)
        val_loader = torch.utils.data.DataLoader(
            validation_data,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True)
        val_logger = Logger(
            os.path.join(opt.result_path, 'val.log'), ['epoch', 'loss', 'acc'])

    if opt.resume_path:
        print('loading checkpoint {}'.format(opt.resume_path))
        checkpoint = torch.load(opt.resume_path)
        assert opt.arch == checkpoint['arch']

        opt.begin_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if not opt.no_train:
            optimizer.load_state_dict(checkpoint['optimizer'])

    # initial selection if necessary
    if opt.pretrain_path and len(train_loader.dataset.indices) == 0:
        print 'initial data selection'
        acqusition(pool_loader, train_loader, model, opt)

    cycle_val_acc = []
    while len(training_data) <= opt.max_train_size:
        print('=========================================')
        print 'train dataset size: ', len(training_data)
        print 'pool  dataset size: ', len(pool_data)
        print 'max train dataset size: ', opt.max_train_size
        print '# pooled data per cycle: ', opt.n_pool

        print('Training')
        max_val_acc = 0
        for i in range(opt.begin_epoch, opt.n_epochs + 1):
            if not opt.no_train:
                train_epoch(i, train_loader, model, criterion, optimizer, opt,
                            train_logger, train_batch_logger)
            if not opt.no_val and i % 50:
                validation_loss, validation_acc = val_epoch(i, val_loader, model, criterion, opt,
                                            val_logger)
                if validation_acc > max_val_acc:
                    max_val_acc = validation_acc

            if not opt.no_train and not opt.no_val:
                scheduler.step(validation_loss)

        cycle_val_acc.append(max_val_acc)
        print cycle_val_acc

        # pool new labeled data
        print('acqusition')
        acqusition(pool_loader, train_loader, model, opt)

        #reset model
        del model
        model, parameters = generate_model(opt)
        # reset optimizer
        optimizer = optim.SGD(
            parameters,
            lr=opt.learning_rate,
            momentum=opt.momentum,
            dampening=dampening,
            weight_decay=opt.weight_decay,
            nesterov=opt.nesterov)
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=opt.lr_patience)
