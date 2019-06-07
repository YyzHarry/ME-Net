from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import numpy as np
import os
import sys
from PIL import Image
from cvxpy import *
from fancyimpute import SoftImpute, BiScaler

import tensorflow as tf
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as Data
import foolbox

from cleverhans.attacks import SPSA
from cleverhans.model import CallableModelWrapper
from cleverhans.utils_pytorch import convert_pytorch_model_to_tf

import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()
# Directory
parser.add_argument('--data-dir', default='./data/', help='data path')
parser.add_argument('--ckpt-dir', default='./checkpoint/', help='checkpoint path')
parser.add_argument('--name', type=str, default='0', help='name of saved checkpoints')
# ME parameters
parser.add_argument('--me-channel', type=str, default='concat',
                    choices=['separate', 'concat'],
                    help='handle RGB channels separately as independent matrices, or jointly by concatenating')
parser.add_argument('--me-type', type=str, default='usvt',
                    choices=['usvt', 'softimp', 'nucnorm'],
                    help='method of matrix estimation')
# Hyper-parameters
parser.add_argument('--mu', type=float, default=1, help='Nuclear Norm hyper-param (default: 1)')
parser.add_argument('--svdprob', type=float, default=0.8, help='USVT hyper-param (default: 0.8)')
parser.add_argument('--maskp', type=float, default=0.5, help='probability of mask sampling (default: 0.5)')
# Attack parameters
parser.add_argument('--source', type=str, default='baseline',
                    help='source model of transfer-based black-box attacks')
parser.add_argument('--attack-type', type=str, default='pgd',
                    choices=['fgsm', 'pgd', 'cw', 'spsa', 'boundary'])
parser.add_argument('--epsilon', type=float, default=8, help='The upper bound change of L-inf norm on input pixels')
parser.add_argument('--iter', type=int, default=1000, help='The number of iterations for iterative attacks')
parser.add_argument('--cw-conf', type=int, default=20, help='The confidence of adversarial examples for CW attack')
parser.add_argument('--spsa-sample', type=int, default=2048, help='The number of SPSA samples for SPSA attack')

args = parser.parse_args()


config = {
    'epsilon': args.epsilon / 255.,
    'num_steps': args.iter,
    'step_size': 2.0 / 255,
    'random_start': True,
}


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Normalization param
mean = np.array([0.4914, 0.4822, 0.4465]).reshape((3, 1, 1))
std = np.array([0.2023, 0.1994, 0.2010]).reshape((3, 1, 1))


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def get_data():
    batch = unpickle(args.data_dir + 'cifar-10-batches-py/test_batch')
    data = batch[b'data']
    labels = batch[b'labels']
    return data, labels


def target_transform(label):
    label = np.array(label)
    target = torch.from_numpy(label).long()
    return target


class CIFAR10_testset(Data.Dataset):

    def __init__(self, target_transform=None):
        self.target_transform = target_transform
        self.test_data, self.test_labels = get_data()
        self.test_data = self.test_data.reshape((10000, 3, 32, 32))
        self.test_data = self.test_data.transpose((0, 2, 3, 1))

    def __getitem__(self, index):
        img, label = self.test_data[index], self.test_labels[index]

        img = Image.fromarray(img)
        img = transform_test(img)
        if self.target_transform is not None:
            target = self.target_transform(label)

        return img, target

    def __len__(self):
        return len(self.test_data)


def nuclear_norm_solve(A, mask, mu):
    """Nuclear norm minimization solver.

    :param A: matrix to complete
    :param mask: matrix with entries zero (if missing) or one (if present)
    :param mu: control trade-off between nuclear norm and square loss
    :return: completed matrix
    """
    X = Variable(shape=A.shape)
    objective = Minimize(mu * norm(X, "nuc") + sum_squares(multiply(mask, X-A)))
    problem = Problem(objective, [])
    problem.solve(solver=SCS)
    return X.value


class nucnorm(torch.autograd.Function):
    """ME-Net layer with nuclear norm algorithm.

    The ME preprocessing is embedded into a Function subclass for adversarial training.
    ----------
    Candès, J. and Recht, B. Exact matrix completion via convex optimization. 2009.
    https://pytorch.org/docs/stable/notes/extending.html
    """

    @staticmethod
    def forward(ctx, input):
        batch_num, c, h, w = input.size()
        output = torch.zeros_like(input).cpu().numpy()

        for i in range(batch_num):
            img = (input[i] * 2 - 1).cpu().numpy()

            if args.me_channel == 'concat':
                img = np.concatenate((np.concatenate((img[0], img[1]), axis=1), img[2]), axis=1)
                mask = np.random.binomial(1, args.maskp, h * w * c).reshape(h, w * c)
                W = nuclear_norm_solve(img, mask, mu=args.mu)
                W[W < -1] = -1
                W[W > 1] = 1
                est_matrix = (W + 1) / 2
                for channel in range(c):
                    output[i, channel] = est_matrix[:, channel * h:(channel + 1) * h]
            else:
                mask = np.random.binomial(1, args.maskp, h * w).reshape(h, w)
                for channel in range(c):
                    W = nuclear_norm_solve(img[channel], mask, mu=args.mu)
                    W[W < -1] = -1
                    W[W > 1] = 1
                    output[i, channel] = (W + 1) / 2

        output = output - mean
        output /= std
        output = torch.from_numpy(output).float().to(device)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # BPDA, approximate gradients
        return grad_output


class usvt(torch.autograd.Function):
    """ME-Net layer with universal singular value thresholding (USVT) approach.

    The ME preprocessing is embedded into a Function subclass for adversarial training.
    ----------
    Chatterjee, S. et al. Matrix estimation by universal singular value thresholding. 2015.
    https://pytorch.org/docs/stable/notes/extending.html
    """

    @staticmethod
    def forward(ctx, input):
        batch_num, c, h, w = input.size()
        output = torch.zeros_like(input).cpu().numpy()

        for i in range(batch_num):
            img = (input[i] * 2 - 1).cpu().numpy()

            if args.me_channel == 'concat':
                img = np.concatenate((np.concatenate((img[0], img[1]), axis=1), img[2]), axis=1)
                mask = np.random.binomial(1, args.maskp, h * w * c).reshape(h, w * c)
                p_obs = len(mask[mask == 1]) / (h * w * c)

                u, sigma, v = np.linalg.svd(img * mask)
                S = np.zeros((h, w))
                for j in range(int(args.svdprob * h)):
                    S[j][j] = sigma[j]
                S = np.concatenate((S, np.zeros((h, w * 2))), axis=1)
                W = np.dot(np.dot(u, S), v) / p_obs
                W[W < -1] = -1
                W[W > 1] = 1
                est_matrix = (W + 1) / 2
                for channel in range(c):
                    output[i, channel] = est_matrix[:, channel * h:(channel + 1) * h]
            else:
                mask = np.random.binomial(1, args.maskp, h * w).reshape(h, w)
                p_obs = len(mask[mask == 1]) / (h * w)
                for channel in range(c):
                    u, sigma, v = np.linalg.svd(img[channel] * mask)
                    S = np.zeros((h, w))
                    for j in range(int(args.svdprob * h)):
                        S[j][j] = sigma[j]
                    W = np.dot(np.dot(u, S), v) / p_obs
                    W[W < -1] = -1
                    W[W > 1] = 1
                    output[i, channel] = (W + 1) / 2

        output = output - mean
        output /= std
        output = torch.from_numpy(output).float().to(device)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # BPDA, approximate gradients
        return grad_output


class softimp(torch.autograd.Function):
    """ME-Net layer with Soft-Impute approach.

    The ME preprocessing is embedded into a Function subclass for adversarial training.
    ----------
    Mazumder, R. et al. Spectral regularization algorithms for learning large incomplete matrices. 2010.
    https://pytorch.org/docs/stable/notes/extending.html
    """

    @staticmethod
    def forward(ctx, input):
        batch_num, c, h, w = input.size()
        output = torch.zeros_like(input).cpu().numpy()

        for i in range(batch_num):
            img = (input[i] * 2 - 1).cpu().numpy()

            if args.me_channel == 'concat':
                img = np.concatenate((np.concatenate((img[0], img[1]), axis=1), img[2]), axis=1)
                mask = np.random.binomial(1, args.maskp, h * w * c).reshape(h, w * c).astype(float)
                mask[mask < 1] = np.nan
                W = SoftImpute(verbose=False).fit_transform(mask * img)
                W[W < -1] = -1
                W[W > 1] = 1
                est_matrix = (W + 1) / 2
                for channel in range(c):
                    output[i, channel] = est_matrix[:, channel * h:(channel + 1) * h]
            else:
                mask = np.random.binomial(1, args.maskp, h * w).reshape(h, w).astype(float)
                mask[mask < 1] = np.nan
                for channel in range(c):
                    mask_img = img[channel] * mask
                    W = SoftImpute(verbose=False).fit_transform(mask_img)
                    W[W < -1] = -1
                    W[W > 1] = 1
                    output[i, channel] = (W + 1) / 2

        output = output - mean
        output /= std
        output = torch.from_numpy(output).float().to(device)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # BPDA, approximate gradients
        return grad_output


class MENet(nn.Module):
    """ME-Net layer.

    To attack a trained ME-Net model, first load the checkpoint, then wrap the loaded model with ME layer.
    Example:
        model = checkpoint['model']
        menet_model = MENet(model)
    ----------
    https://pytorch.org/docs/stable/notes/extending.html
    """
    def __init__(self, model):
        super(MENet, self).__init__()
        self.model = model

    def forward(self, input):
        x = globals()[args.me_type].apply(input)
        return self.model(x)


def transfer_attack():
    fmodel = foolbox.models.PyTorchModel(menet_model, bounds=(0, 1), num_classes=10, preprocessing=(0, 1))
    fmodel_source = foolbox.models.PyTorchModel(source_model, bounds=(0, 1), num_classes=10, preprocessing=(0, 1))
    attack_criteria = foolbox.criteria.Misclassification()

    if args.attack_type == 'fgsm':
        attack = foolbox.attacks.GradientSignAttack(model=fmodel_source, criterion=attack_criteria)
    elif args.attack_type == 'pgd':
        attack = foolbox.attacks.ProjectedGradientDescentAttack(model=fmodel_source, criterion=attack_criteria)
    else:
        attack = foolbox.attacks.CarliniWagnerL2Attack(model=fmodel_source, criterion=attack_criteria)

    correct = 0
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.cpu().numpy()[0], int(targets.cpu().numpy())

        if args.attack_type == 'fgsm':
            adversarial = attack(inputs.astype(np.float32), targets, max_epsilon=config['epsilon'])
        elif args.attack_type == 'pgd':
            adversarial = attack(inputs.astype(np.float32), targets, epsilon=config['epsilon'],
                                 stepsize=config['step_size'], iterations=config['num_steps'])
        else:
            adversarial = attack(inputs.astype(np.float32), targets, max_iterations=config['num_steps'],
                                 confidence=args.cw_conf)
        if adversarial is None:
            adversarial = inputs.astype(np.float32)
        if np.argmax(fmodel.predictions(adversarial)) == targets:
            correct += 1.

        sys.stdout.write("\rTransfer-based black-box %s attack... Acc: %.3f%% (%d/%d)" %
                         (args.attack_type, 100. * correct / (batch_idx + 1), correct, batch_idx + 1))
        sys.stdout.flush()

    print('Accuracy under transfer-based %s attack: %.3f%%' % (args.attack_type, 100. * correct / batch_idx))


def boundary_attack():
    fmodel = foolbox.models.PyTorchModel(menet_model, bounds=(0, 1), num_classes=10, preprocessing=(0, 1))
    attack_criteria = foolbox.criteria.Misclassification()
    attack = foolbox.attacks.BoundaryAttack(model=fmodel, criterion=attack_criteria)

    correct = 0
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.cpu().numpy()[0], int(targets.cpu().numpy())
        adversarial = attack(inputs.astype(np.float32), targets, iterations=args.iter, log_every_n_steps=999999)
        if adversarial is None:
            adversarial = inputs.astype(np.float32)
        if np.argmax(fmodel.predictions(adversarial)) == targets:
            correct += 1.

        sys.stdout.write("\rBlack-box Boundary attack... Acc: %.3f%% (%d/%d)" %
                         (100. * correct / (batch_idx + 1), correct, batch_idx + 1))
        sys.stdout.flush()

    print('Accuracy under Boundary attack: %.3f%%' % (100. * correct / batch_idx))


def spsa_attack():
    # Use tf for evaluation on adversarial data
    sess = tf.Session()
    x_op = tf.placeholder(tf.float32, shape=(None, 3, 32, 32,))
    y_op = tf.placeholder(tf.float32, shape=(1,))

    # Convert pytorch model to a tf_model and wrap it in cleverhans
    tf_model_fn = convert_pytorch_model_to_tf(menet_model)
    cleverhans_model = CallableModelWrapper(tf_model_fn, output_layer='logits')

    # Create an SPSA attack
    spsa = SPSA(cleverhans_model, sess=sess)
    spsa_params = {
        'eps': config['epsilon'],
        'nb_iter': config['num_steps'],
        'clip_min': 0.,
        'clip_max': 1.,
        'spsa_samples': args.spsa_sample,  # in this case, the batch_size is equal to spsa_samples
        'spsa_iters': 1,
    }

    adv_x_op = spsa.generate(x_op, y_op, **spsa_params)
    adv_preds_op = tf_model_fn(adv_x_op)

    # Evaluation against SPSA attacks
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        adv_preds = sess.run(adv_preds_op, feed_dict={x_op: inputs, y_op: targets})
        correct += (np.argmax(adv_preds, axis=1) == targets).sum().float()
        total += len(inputs)

        sys.stdout.write(
            "\rBlack-box SPSA attack... Acc: %.3f%% (%d/%d)" % (100. * correct / total, correct, total))
        sys.stdout.flush()

    print('Accuracy under SPSA attack: %.3f%%' % (100. * correct / total))


if __name__ == '__main__':

    print('=====> Preparing data...')
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    test_dataset = CIFAR10_testset(target_transform)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=4)

    # Load checkpoint
    print('=====> Loading trained model from checkpoint...')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(args.ckpt_dir + args.name + '.ckpt')

    model = checkpoint['model']
    rng_state = checkpoint['rng_state']
    torch.set_rng_state(rng_state)

    model.eval()
    model = model.to(device)
    menet_model = MENet(model)
    menet_model.eval()

    # Baseline
    checkpoint = torch.load(args.ckpt_dir + args.source + '.ckpt')
    source = checkpoint['model']
    source.eval()
    source = source.to(device)
    source_model = MENet(source)
    source_model.eval()

    # score-based attack
    if args.attack_type == 'spsa':
        spsa_attack()
    # decision-based attack
    elif args.attack_type == 'boundary':
        boundary_attack()
    # transfer-based attack
    else:
        transfer_attack()
