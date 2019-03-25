import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from util.transforms import Identity
import logging
import torch

FORMAT = '[%(asctime)-15s %(filename)s:%(lineno)s] %(message)s'
FORMAT_MINIMAL = '%(message)s'

logger = logging.getLogger('neuralODE')
logging.basicConfig(format=FORMAT)
logger.setLevel(logging.DEBUG)


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


def inf_generator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


def learning_rate_with_decay(
        lr,
        batch_size,
        batch_denom,
        batches_per_epoch,
        boundary_epochs,
        decay_rates):
    initial_learning_rate = lr*1. #* batch_size / batch_denom

    boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
    vals = [initial_learning_rate * decay for decay in decay_rates]

    def learning_rate_fn(itr):
        lt = [itr < b for b in boundaries] + [True]
        i = np.argmax(lt)
        return vals[i]

    return learning_rate_fn


def one_hot(x, K):
    return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)


def accuracy(model, dataset_loader, args):
    total_correct = 0
    for x, y in dataset_loader:
        if args.gpu:
            x = x.cuda()
        y = one_hot(np.array(y.numpy()), 10)

        target_class = np.argmax(y, axis=1)
        predicted_class = np.argmax(model(x).cpu().detach().numpy(), axis=1)
        total_correct += np.sum(predicted_class == target_class)
    return total_correct / len(dataset_loader.dataset)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def flatten(grad_calculator):
    def wrapper():
        return torch.cat([grad.view(-1) for grad in grad_calculator()])

    return wrapper


def get_dataset(name='cifar10', tensor_type_transformer=Identity):
    """
        return: train_dataset and test_dataset.
    """


    if name == 'mnist':
        num_classes = 10
        train_dataset = datasets.MNIST(
            '../data',
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                tensor_type_transformer(),
            ]))
        test_dataset = datasets.MNIST(
            '../data',
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                tensor_type_transformer(),
            ]))
    if name == 'svhn':
        num_classes = 10
        train_dataset = datasets.SVHN(
            '../data',
            split='extra',
            download=True,
            transform=transforms.Compose([transforms.ToTensor()]))
        test_dataset = datasets.SVHN(
            '../data',
            split='test',
            download=True,
            transform=transforms.Compose([transforms.ToTensor()]))

    if name == 'cifar10':
        num_classes = 10
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            tensor_type_transformer(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            tensor_type_transformer(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        train_dataset = datasets.CIFAR10(
            root='../data',
            train=True,
            download=True,
            transform=transform_train)

        test_dataset = datasets.CIFAR10(
            root='../data',
            train=False,
            download=False,
            transform=transform_test)

    if name == 'cifar100':

        num_classes = 100
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            tensor_type_transformer(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            tensor_type_transformer(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        train_dataset = datasets.CIFAR100(
            root='../data',
            train=True,
            download=True,
            transform=transform_train)

        test_dataset = datasets.CIFAR100(
            root='../data',
            train=False,
            download=False,
            transform=transform_test)

    if name == 'tinyimagenet':
        num_classes = 200
        normalize = transforms.Normalize(
            mean=[
                0.44785526394844055, 0.41693055629730225, 0.36942949891090393
            ],
            std=[0.2928885519504547, 0.28230994939804077, 0.2889912724494934])
        train_dataset = datasets.ImageFolder(
            '../data/tiny-imagenet-200/train',
            transforms.Compose([
                transforms.RandomCrop(64, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        test_dataset = datasets.ImageFolder(
            '../data/tiny-imagenet-200/val',
            transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]))

    return train_dataset, test_dataset, num_classes


def get_data(name='cifar10', train_bs=128, test_bs=1000):
    """
        return: train_data_loader, test_data_loader
    """
    train_dataset, test_dataset = get_dataset(name)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_bs, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_bs, shuffle=False)
    return train_loader, test_loader


def exp_lr_scheduler(epoch,
                     optimizer,
                     strategy=True,
                     decay_eff=0.1,
                     decayEpoch=[]):
    """Decay learning rate by a factor of lr_decay every lr_decay_epoch epochs"""

    if strategy == 'normal':
        if epoch in decayEpoch:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= decay_eff
    else:
        print('wrong strategy')
        raise ValueError('A very specific bad thing happened.')

    return optimizer


def lr_calculator(epoch,
                  current_lr,
                  strategy=True,
                  decay_eff=0.1,
                  decayEpoch=[]):

    if strategy == 'normal':
        if epoch in decayEpoch:
            return current_lr * decay_eff
        else: 
            return current_lr
    else:
        print('wrong strategy')
        raise ValueError('A very specific bad thing happened.')


def fb_warmup(optimizer, epoch, baselr, large_ratio):
    for param_group in optimizer.param_groups:
        param_group['lr'] = epoch * \
            (baselr * large_ratio - baselr) / 5. + baselr
    return optimizer


# useful
def group_add(params, update, lmbd=1):
    """
    params = params + update*alpha
    :param params: list of variable
    :param update: list of data
    :return:
    """
    for i, p in enumerate(params):
        params[i].add_(update[i] * lmbd + 0.)
    return params


def group_product(xs, ys):
    """
    the inner product of two lists of variables xs,ys
    :param xs:
    :param ys:
    :return:
    """
    return sum([torch.sum(x * y) for (x, y) in zip(xs, ys)])


def get_p_g_m(opt, layers):
    i = 0
    paramlst = []
    grad = []
    mum = []

    for group in opt.param_groups:
        weight_decay = group['weight_decay']
        momentum = group['momentum']

        for p in group['params']:
            if p.grad is None:
                continue
            p.grad.data = p.grad.data
            d_p = p.grad.data
            if momentum != 0:
                param_state = opt.state[p]
                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = torch.zeros_like(
                        p.data)
                else:
                    buf = param_state['momentum_buffer']
            if i in layers:
                paramlst.append(p.data)
                grad.append(d_p + 0.)  # get grad
                mum.append(buf * momentum + 0.)
            i += 1
    return paramlst, grad, mum


def manually_update(opt, grad):
    for group in opt.param_groups:
        weight_decay = group['weight_decay']
        momentum = group['momentum']
        dampening = group['dampening']
        nesterov = group['nesterov']

        for i, p in enumerate(group['params']):
            d_p = grad[i]
            if weight_decay != 0:
                d_p.add_(weight_decay, p.data)
            if momentum != 0:
                param_state = opt.state[p]
                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = torch.zeros_like(
                        p.data)
                    buf.mul_(momentum).add_(d_p)
                else:
                    buf = param_state['momentum_buffer']
                    buf.mul_(momentum).add_(1 - dampening, d_p)
                if nesterov:
                    d_p = d_p.add(momentum, buf)
                else:
                    d_p = buf
            p.data.add_(-group['lr'], d_p)


def change_lr_single(optimizer, best_lr):
    """change learning rate"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = best_lr
    return optimizer


def bad_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal(m.weight, std=0.5)
            if m.bias is not None:
                nn.init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal(m.weight, std=0.5)
            if m.bias is not None:
                nn.init.constant(m.bias, 0)
