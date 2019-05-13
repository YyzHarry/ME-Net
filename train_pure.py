from __future__ import print_function

import argparse
import numpy as np
import os
import csv
import math
from PIL import Image
from cvxpy import *
from fancyimpute import SoftImpute, BiScaler

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as Data

import models
from utils import progress_bar


# Checkpoint related
START_EPOCH = 0

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def nuclear_norm_solve(A, mask, mu):
    X = Variable(shape=A.shape)
    objective = Minimize(mu * norm(X, "nuc") + sum_squares(multiply(mask, X-A)))
    problem = Problem(objective, [])
    problem.solve(solver=SCS)
    return X.value


def nucnorm(img, maskp):
    h, w, c = img.shape
    img = img.astype('float64') * 2 / 255 - 1

    if args.me_channel == 'concat':
        img = img.transpose(2, 0, 1)
        img = np.concatenate((np.concatenate((img[0], img[1]), axis=1), img[2]), axis=1)
        mask = np.random.binomial(1, maskp, h * w * c).reshape(h, w * c)
        W = nuclear_norm_solve(img, mask, mu=args.mu)
        W[W < -1] = -1
        W[W > 1] = 1
        est_matrix = (W + 1) * 255 / 2
        outputs = np.zeros((h, w, c))
        for channel in range(c):
            outputs[:, :, channel] = est_matrix[:, channel * w:(channel + 1) * w]
    else:
        mask = np.random.binomial(1, maskp, h * w).reshape(h, w)
        outputs = np.zeros((h, w, c))
        for channel in range(c):
            W = nuclear_norm_solve(img[:, :, channel], mask, mu=args.mu)
            W[W < -1] = -1
            W[W > 1] = 1
            outputs[:, :, channel] = (W + 1) * 255 / 2

    return outputs


def usvt(img, maskp):
    h, w, c = img.shape
    img = img.astype('float64') * 2 / 255 - 1

    if args.me_channel == 'concat':
        img = img.transpose(2, 0, 1)
        img = np.concatenate((np.concatenate((img[0], img[1]), axis=1), img[2]), axis=1)
        mask = np.random.binomial(1, maskp, h * w * c).reshape(h, w * c)
        p_obs = len(mask[mask == 1]) / (h * w * c)

        u, sigma, v = np.linalg.svd(img * mask)
        S = np.zeros((h, h))
        for j in range(int(args.svdprob * h)):
            S[j][j] = sigma[j]
        S = np.concatenate((S, np.zeros((h, w*(c-1)))), axis=1)

        W = np.dot(np.dot(u, S), v) / p_obs
        W[W < -1] = -1
        W[W > 1] = 1
        est_matrix = (W + 1) * 255 / 2
        outputs = np.zeros((h, w, c))
        for channel in range(c):
            outputs[:, :, channel] = est_matrix[:, channel * w:(channel + 1) * w]
    else:
        mask = np.random.binomial(1, maskp, h * w).reshape(h, w)
        p_obs = len(mask[mask == 1]) / (h * w)

        outputs = np.zeros((h, w, c))
        for channel in range(c):
            u, sigma, v = np.linalg.svd(img[:, :, channel] * mask)
            S = np.zeros((h, h))
            sigma = np.concatenate((sigma, np.zeros(h - len(sigma))), axis=0)
            for j in range(int(args.svdprob * h)):
                S[j][j] = sigma[j]

            W = np.dot(np.dot(u, S), v) / p_obs
            W[W < -1] = -1
            W[W > 1] = 1
            outputs[:, :, channel] = (W + 1) * 255 / 2

    return outputs


def softimp(img, maskp):
    h, w, c = img.shape
    img = img.astype('float64') * 2 / 255 - 1

    if args.me_channel == 'concat':
        img = img.transpose(2, 0, 1)
        img = np.concatenate((np.concatenate((img[0], img[1]), axis=1), img[2]), axis=1)
        mask = np.random.binomial(1, maskp, h * w * c).reshape(h, w * c).astype(float)
        mask[mask < 1] = np.nan

        W = SoftImpute(verbose=False).fit_transform(mask * img)
        W[W < -1] = -1
        W[W > 1] = 1
        est_matrix = (W + 1) * 255 / 2
        outputs = np.zeros((h, w, c))
        for channel in range(c):
            outputs[:, :, channel] = est_matrix[:, channel * w:(channel + 1) * w]
    else:
        mask = np.random.binomial(1, maskp, h * w).reshape(h, w).astype(float)
        mask[mask < 1] = np.nan

        outputs = np.zeros((h, w, c))
        for channel in range(c):
            mask_img = img[:, :, channel] * mask
            W = SoftImpute(verbose=False).fit_transform(mask_img)
            W[W < -1] = -1
            W[W > 1] = 1
            outputs[:, :, channel] = (W + 1) * 255 / 2

    return outputs


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def get_data(train=False):
    data = None
    labels = None
    if train:
        for i in range(1, 6):
            batch = unpickle(args.data_dir + 'cifar-10-batches-py/data_batch_' + str(i))
            if i == 1:
                data = batch[b'data']
            else:
                data = np.concatenate([data, batch[b'data']])
            if i == 1:
                labels = batch[b'labels']
            else:
                labels = np.concatenate([labels, batch[b'labels']])

        data_tmp = data
        labels_tmp = labels
        # repeat n times for different masks
        for i in range(args.mask_num - 1):
            data = np.concatenate([data, data_tmp])
            labels = np.concatenate([labels, labels_tmp])
    else:
        batch = unpickle(args.data_dir + 'cifar-10-batches-py/test_batch')
        data = batch[b'data']
        labels = batch[b'labels']
    return data, labels


def target_transform(label):
    label = np.array(label)
    target = torch.from_numpy(label).long()
    return target


# ME-Net pre-processing
def menet(train_data, train=True):
    if train:
        for i in range(train_data.shape[0]):
            maskp = args.startp + math.ceil((i + 1) / 50000) * (args.endp - args.startp) / args.mask_num
            train_data[i] = globals()[args.me_type](train_data[i], maskp)
            # Bar visualization
            progress_bar(i, train_data.shape[0], ' | Training data')

    else:
        for i in range(train_data.shape[0]):
            maskp = (args.startp + args.endp) / 2
            train_data[i] = globals()[args.me_type](train_data[i], maskp)
            # Bar visualization
            progress_bar(i, train_data.shape[0], ' | Testing data')

    return train_data


class CIFAR10_Dataset(Data.Dataset):

    def __init__(self, train=True, target_transform=None):
        self.target_transform = target_transform
        self.train = train

        if self.train:
            self.train_data, self.train_labels = get_data(train)
            self.train_data = self.train_data.reshape((self.train_data.shape[0], 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))
            self.train_data = menet(self.train_data, train=True)
        else:
            self.test_data, self.test_labels = get_data()
            self.test_data = self.test_data.reshape((10000, 3, 32, 32))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))
            self.test_data = menet(self.test_data, train=False)

    def __getitem__(self, index):
        if self.train:
            img, label = self.train_data[index], self.train_labels[index]
        else:
            img, label = self.test_data[index], self.test_labels[index]

        img = Image.fromarray(img)
        if self.train:
            img = transform_train(img)
        else:
            img = transform_test(img)
        if self.target_transform is not None:
            target = self.target_transform(label)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)


def train(epoch):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        train_loss += loss.item()
        _, pred_idx = torch.max(outputs.data, 1)

        total += targets.size(0)
        correct += pred_idx.eq(targets.data).cpu().sum().float()

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        progress_bar(batch_idx, len(train_loader),
                     'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    return train_loss/batch_idx, 100.*correct/total


def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, pred_idx = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += pred_idx.eq(targets.data).cpu().sum().float()

        progress_bar(batch_idx, len(test_loader),
                     'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    return test_loss/batch_idx, 100.*correct/total


def save_checkpoint(acc, epoch):
    print('=====> Saving checkpoint...')
    state = {
        'model': model,
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, args.save_dir + args.name + '_epoch' + str(epoch) + '.ckpt')


# Decrease the learning rate at 100 and 150 epoch
def adjust_lr(optimizer, epoch):
    lr = args.lr
    if epoch >= 100:
        lr /= 10
    if epoch >= 150:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # Directory
    parser.add_argument('--data-dir', default='./data/', help='data path')
    parser.add_argument('--save-dir', default='./checkpoint/', help='save path')
    # Hyper-parameters
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate (default=0.1)')
    parser.add_argument('--mu', type=float, default=1, help='Nuclear Norm hyper-param (default: 1)')
    parser.add_argument('--svdprob', type=float, default=0.8, help='USVT hyper-param (default: 0.8)')
    parser.add_argument('--startp', type=float, default=0.8, help='start probability of mask sampling (default: 0.8)')
    parser.add_argument('--endp', type=float, default=1, help='end probability of mask sampling (default: 1)')
    parser.add_argument('--batch-size', '-b', type=int, default=256, help='batch size')
    parser.add_argument('--epoch', type=int, default=200, help='total epochs')
    parser.add_argument('--no-augment', dest='augment', action='store_false')
    parser.add_argument('--decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--mask-num', type=int, default=10, help='number of sampled masks (default: 10)')
    parser.add_argument('--num_ckpt_steps', type=int, default=10, help='save checkpoint steps (default: 10)')
    # ME parameters
    parser.add_argument('--me-channel', type=str, default='concat',
                        choices=['separate', 'concat'],
                        help='handle RGB channels separately as independent matrices, or jointly by concatenating')
    parser.add_argument('--me-type', type=str, default='usvt',
                        choices=['usvt', 'softimp', 'nucnorm'],
                        help='method of matrix estimation')
    # Utility parameters
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--model', type=str, default='ResNet18', help='choose model type (default: ResNet18)')
    parser.add_argument('--name', type=str, help='name of the run')

    args = parser.parse_args()

    # Data
    print('=====> Preparing data...')

    if args.augment:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = CIFAR10_Dataset(True, target_transform)
    test_dataset = CIFAR10_Dataset(False, target_transform)

    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        batch_size = args.batch_size * n_gpu

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=6*n_gpu)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=6*n_gpu)

    # Models
    if args.resume:
        print('=====> Resuming from checkpoint...')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.save_dir + args.name + '.ckpt')

        model = checkpoint['model']
        acc = checkpoint['acc']
        START_EPOCH = checkpoint['epoch'] + 1
        rng_state = checkpoint['rng_state']
        torch.set_rng_state(rng_state)
    else:
        print('=====> Building model...')
        model = models.__dict__[args.model]()

    model = model.to(device)

    if not os.path.isdir('results'):
        os.mkdir('results')
    logname = ('results/log_' + args.name + '.csv')

    if torch.cuda.device_count() > 1:
        print("=====> Use", torch.cuda.device_count(), "GPUs")
        model = nn.DataParallel(model)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.decay)

    if not os.path.exists(logname):
        with open(logname, 'w') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow(['Epoch', 'Train Loss', 'Train Acc', 'Test Loss', 'Test Acc'])

    for epoch in range(START_EPOCH, args.epoch):
        train_loss, train_acc = train(epoch)
        test_loss, test_acc = test(epoch)
        adjust_lr(optimizer, epoch)
        with open(logname, 'a') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow([epoch, train_loss, train_acc, test_loss, test_acc])

        if epoch % args.num_ckpt_steps == 0:
            save_checkpoint(test_acc, epoch)
