import os
import argparse
import logging
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from os.path import dirname, join
from util.utils import get_dataset, RunningAverageMeter, inf_generator, learning_rate_with_decay, one_hot, accuracy, count_parameters
from util.transforms import ToDouble, Identity

parent_dir = join(dirname(__file__), os.pardir)
results_dir = join(parent_dir, 'results')

logger = logging.getLogger('neuralODE')


def add_general_arguments(parser):

    parser.add_argument(
        '--model-prefix',
        '-mp',
        type=str,
        default='',
        required=False,
        metavar='N',
        help='Output file name (default training)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=10,
        metavar='N',
        help='How many outer loops to wait before saving training state')
    parser.add_argument(
        '--epochs',
        type=int,
        default=10000,
        metavar='N',
        help='Number of epochs to train (default: 10000)')
    parser.add_argument(
        '--dataset',
        type=str,
        default='mnist',
        metavar='D',
        help='Dataset (default cifar10)')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=128,
        metavar='N',
        help='Batch size for training (default: 128)')
    parser.add_argument(
        '--test-batch-size',
        type=int,
        default=1000,
        metavar='N',
        help='Batch size for testing (default: 1000)')
    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        metavar='S',
        help='Random seed (default: 1)')
    parser.add_argument(
        '--double', type=bool, default=False, metavar='D', help='')
    parser.add_argument(
        '--load-model',
        type=str,
        default='',
        metavar='M',
        help='Where to load the trained model if the file exists. Empty means it starts from scratch'
    )
    parser.add_argument(
        '--network',
        type=str,
        choices=[
            'resnet',
            'odenet'],
        default='odenet')
    parser.add_argument('--tol', type=float, default=1e-3)
    parser.add_argument(
        '--adjoint',
        type=eval,
        default=False,
        choices=[
            True,
            False])
    parser.add_argument(
        '--downsampling-method',
        type=str,
        default='conv',
        choices=[
            'conv',
            'res'])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument(
        '--lr',
        type=float,
        default=1,
        metavar='LR',
        help='Learning rate (default: 1)')
    parser.add_argument(
        '--lr-decay',
        type=int,
        nargs='+',
        default=[1, 0.1, 0.01],
        help='Decrease learning rate at these epochs.')
    parser.add_argument(
        '--lr-decay-epoch',
        type=int,
        nargs='+',
        default=[30, 60],
        help='Decrease learning rate at these epochs.')
    parser.add_argument(
        '--momentum',
        type=float,
        default=0.9,
        metavar='M',
        help='Momentum (default: 0.9)')
    parser.add_argument(
        '--weight-decay',
        '--wd',
        default=5e-4,
        type=float,
        metavar='W',
        help='Weight decay (default: 1e-4)')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    add_general_arguments(parser)
    args, _ = parser.parse_known_args()

    logger.info(args)

    device = torch.device('cuda:' + str(args.gpu)
                          if torch.cuda.is_available() else 'cpu')

    is_odenet = args.network == 'odenet'

    if args.adjoint:
        from torchdiffeq import odeint_adjoint as odeint
    else:
        from torchdiffeq import odeint

    # Load model. We import here because the default tensor type has been set
    from models import MNIST_model
    model_list = {
        'mnist': MNIST_model(args),
    }
    model, odelayer_index = model_list[args.dataset]
    model = model.to(device)

    logger.info(model)
    logger.info('Number of parameters: {}'.format(count_parameters(model)))

    criterion = nn.CrossEntropyLoss().to(device)

    transformer = ToDouble if args.double else Identity

    train_dataset, test_dataset, num_classes = get_dataset(
        args.dataset, tensor_type_transformer=transformer)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.test_batch_size, shuffle=False)

    data_gen = inf_generator(train_loader)
    batches_per_epoch = len(train_loader)

    lr_fn = learning_rate_with_decay(
        args.lr,
        args.batch_size,
        batch_denom=128,
        batches_per_epoch=batches_per_epoch,
        boundary_epochs=args.lr_decay_epoch,
        decay_rates=args.lr_decay
    )

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    best_acc = 0
    batch_time_meter = RunningAverageMeter()
    f_nfe_meter = RunningAverageMeter()
    b_nfe_meter = RunningAverageMeter()
    end = time.time()

    for itr in range(args.epochs * batches_per_epoch):

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_fn(itr)

        optimizer.zero_grad()
        x, y = data_gen.__next__()
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        if is_odenet:
            nfe_forward = model[odelayer_index].nfe # feature_layers[0].nfe
            #feature_layers[0].nfe = 0
            model[odelayer_index].nfe = 0

        loss.backward()
        optimizer.step()

        if is_odenet:
            nfe_backward = model[odelayer_index].nfe # feature_layers[0].nfe
            # feature_layers[0].nfe = 0
            model[odelayer_index].nfe = 0


        batch_time_meter.update(time.time() - end)
        if is_odenet:
            f_nfe_meter.update(nfe_forward)
            b_nfe_meter.update(nfe_backward)
        end = time.time()

        if itr % batches_per_epoch == 0:
            with torch.no_grad():
                val_acc = accuracy(model, test_loader, device)
                if val_acc > best_acc:
                    torch.save({'state_dict': model.state_dict(), 'args': args}, os.path.join(
                        args.save, 'model.pth'))
                    best_acc = val_acc
                logger.info(
                    "Epoch {:04d} | Time {:.3f} ({:.3f}) | NFE-F {:.1f} | NFE-B {:.1f} | "
                    "Test Acc {:.4f}".format(
                        itr // batches_per_epoch,
                        batch_time_meter.val,
                        batch_time_meter.avg,
                        f_nfe_meter.avg,
                        b_nfe_meter.avg,
                        val_acc))
