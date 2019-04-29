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
from pathlib import Path
import csv

from os.path import dirname, join
from lib.model import *
from lib.trainer import *
from lib.glucose_dataset import *
# from util.utils import get_dataset, RunningAverageMeter, inf_generator, learning_rate_with_decay, one_hot, accuracy, count_parameters
# from util.transforms import ToDouble, Identity

parent_dir = dirname(__file__)
results_dir = join(parent_dir, 'results')
log_dir = join(parent_dir, 'logs')

logging.basicConfig()
logger = logging.getLogger('glucose')
logger.setLevel(logging.INFO)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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
        default=5,
        metavar='N',
        help='How many outer loops to wait before saving training state')
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        metavar='N',
        help='Number of epochs to train (default: 100)')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=256,  # 512 will go out of memory
        metavar='N',
        help='Batch size for training (default: 256)')
    parser.add_argument(
        '--output-len',
        type=int,
        default=6,
        metavar='N',
        help='output length (default: 6)')
    parser.add_argument(
        '--depth',
        type=int,
        default=2,
        metavar='N',
        help='output GRU depth (default: 2)')
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
            'grutorch',  # gru using the torch module
            'gru',       # gru using the self-written module
            'odenet'],   # ODE block
        default='grutorch')
    parser.add_argument('--tol', 
        type=float, 
        default=1e-3)
    parser.add_argument(
        '--adjoint',
        type=eval,
        default=False,
        choices=[
            True,
            False])
    parser.add_argument(
        '--method',
        type=str,
        choices=[
            'explicit_adams', 
            'fixed_adams', 
            'adams',
            'tsit5',
            'dopri5',
            'euler',
            'midpoint',
            'rk4'],
        default='dopri5')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--sequence', type=int, default=1)
    parser.add_argument('--polynomial', type=int, default=0)
    parser.add_argument('--degree', type=int, default=0)
    parser.add_argument(
        '--lr',
        type=float,
        default=0.01,
        metavar='LR',
        help='Learning rate (default: 1)')
    parser.add_argument(
        '--lr-decay',
        type=float,
        default=0.1,
        help='Decrease learning rate at these epochs.')
    parser.add_argument(
        '--lr-decay-epoch',
        type=int,
        default=30,   # lr decah at n*30
        help='Decrease learning rate at these epochs.')
    # parser.add_argument(
    #     '--momentum',
    #     type=float,
    #     default=0.9,
    #     metavar='M',
    #     help='Momentum (default: 0.9)')
    parser.add_argument(
        '--weight-decay',
        '--wd',
        default=1e-5,
        type=float,
        metavar='W',
        help='Weight decay (default: 1e-5)')


def get_file_prefix(args):
    if args.network == 'odenet':
        return "-".join(filter(None, [
            args.model_prefix, 
            args.network, 
            args.method,
            'LR' + str(args.lr),
            # 'momentum' + str(args.momentum),
            'BS' + str(args.batch_size),
            'adjoint' + str(args.adjoint),
            'tol' + str(args.tol),
            'lr-decay' + str(args.lr_decay),
        ]))
    else:
        return "-".join(filter(None, [
            args.model_prefix,
            args.network,
            'LR' + str(args.lr),
            # 'momentum' + str(args.momentum),
            'BS' + str(args.batch_size),
            'lr-decay' + str(args.lr_decay),
        ]))

# def save(epoch, iterations, model, optimizer, args):
#     state = {
#         'epoch': epoch,
#         'iterations': iterations,
#         'model': model.state_dict(),
#         'optimizer': optimizer.state_dict(),
#     }
#     state_string = "-".join(
#         k + "=" + str(v) for k, v in state.items() if k == 'epoch')

#     location = join(results_dir, get_file_prefix(args) + '-s-' + state_string)
#     torch.save(state, location)
#     logger.info(f'Saved to {location}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    add_general_arguments(parser)
    args, _ = parser.parse_known_args()

    # Set up normal logging
    root_logger = logging.getLogger()
    log_path = join(results_dir, f'{get_file_prefix(args)}.log')
    file_handler = logging.FileHandler(log_path)
    root_logger.addHandler(file_handler)

    for arg in vars(args):
        logger.info(f'{arg} {getattr(args, arg)}')

    device = torch.device('cuda:' + str(args.gpu)
                          if torch.cuda.is_available() else 'cpu')

    is_odenet = args.network == 'odenet'

    if args.adjoint:
        from torchdiffeq import odeint_adjoint as odeint
    else:
        from torchdiffeq import odeint

    # Set up CSV logging
    csv_path = join(results_dir, f'{get_file_prefix(args)}.csv')
    is_new_log = not Path(csv_path).exists()
    csv_file = open(csv_path, 'a', newline='')
    writer = csv.writer(
        csv_file, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    if is_new_log:
        writer.writerow([
            'epoch', 'train_loss', 'valid_loss'
        ])

    # import the model 
    model = MultiOutputRNN(input_dim=1,
                            output_dim=361,
                            hidden_size=512,
                            output_len=args.output_len,
                            depth=args.depth,
                            cuda=args.gpu,
                            autoregressive=False,
                            args = args,)

    logger.info(model)
    logger.info('Number of parameters: {}'.format(count_parameters(model)))

    train_data = GlucoseDataset(data_pkl='data/processed_cgm_data_train.pkl',
                                max_pad=101,
                                output_len=args.output_len, 
                                output_dim=361,
                                polynomial=args.polynomial,
                                degree=args.degree,
                                range_low=0,
                                range_high=100)

    val_data = GlucoseDataset(data_pkl='data/processed_cgm_data_validation.pkl',
                                max_pad=101,
                                output_len=args.output_len,
                                output_dim=361,
                                polynomial=args.polynomial,
                                degree=args.degree,
                                range_low=0,
                                range_high=100)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        # momentum=args.momentum,
        weight_decay=args.weight_decay)

    criterion = nn.CrossEntropyLoss().to(device)

    path = join(results_dir, f'{get_file_prefix(args)}')

    trainer = ExperimentTrainer(model=model, 
                                optimizer=optimizer, 
                                criterion=criterion, 
                                name=path, 
                                model_dir=results_dir, 
                                log_dir=log_dir,
                                load=False, 
                                load_epoch=None,
                                cuda=args.gpu,
                                csv_writer=writer,
                                csv_file=csv_file,
                                lr_decay = args.lr_decay,
                                lr_decay_epoch = args.lr_decay_epoch)

    trainer.train_sup(epoch_lim=args.epochs, 
                        data=train_data, 
                        valid_data=val_data, 
                        early_stopping_lim=None,
                        batch_size=args.batch_size, 
                        num_workers=1, 
                        track_embeddings=False, 
                        validation_rate=args.save_interval, 
                        loss_weight_base=1,
                        value_weight=0, 
                        value_ratio=0)

    # # Initialize optimizer 
    # lr_fn = learning_rate_with_decay(
    #     args.lr,
    #     args.batch_size,
    #     batch_denom=128,
    #     batches_per_epoch=batches_per_epoch,
    #     boundary_epochs=args.lr_decay_epoch,
    #     decay_rates=args.lr_decay
    # )


    # best_acc = 0
    # batch_time_meter = RunningAverageMeter()
    # f_nfe_meter = RunningAverageMeter()
    # b_nfe_meter = RunningAverageMeter()
    # end = time.time()


    # # Create/load state
    # epoch = 0
    # iterations = 0
    # if args.load_model is not '':
    #     try:
    #         state_file = args.load_model
    #         state = torch.load(state_file)

    #         epoch = state['epoch']
    #         iterations = state['iterations']

    #         model.load_state_dict(state['model'])

    #         logger.info(f'Loaded state from file {state_file}')
    #     except FileNotFoundError:
    #         raise FileNotFoundError(f'No model exist on: {args.load_model}')

    # train_loss = 0.0

    # logger.info(f'Numer of batches per epoch: {batches_per_epoch}')
    # while iterations < (args.epochs * batches_per_epoch):

    #     logger.info(f'Iteration number: {iterations}')

    #     iterations += 1

    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = lr_fn(iterations)

    #     optimizer.zero_grad()
    #     x, y = data_gen.__next__()
    #     if args.gpu:
    #         x = x.cuda()
    #         y = y.cuda()
    #     logits = model(x)
    #     loss = criterion(logits, y)
    #     train_loss += loss

    #     if is_odenet:
    #         nfe_forward = 0.
    #         for index in odelayer_indexes:
    #             nfe_forward += model[index].nfe
    #             model[index].nfe = 0 
    #         nfe_forward = nfe_forward / len(odelayer_indexes)
    #         logger.info(f'nfe_forward is: {nfe_forward}')

    #     loss.backward()
    #     optimizer.step()

    #     if is_odenet:
    #         nfe_backward = 0.
    #         for index in odelayer_indexes:
    #             nfe_backward += model[index].nfe
    #             model[index].nfe = 0 
    #         nfe_backward = nfe_backward / len(odelayer_indexes)
    #         logger.info(f'nfe_backward is: {nfe_backward}')

    #     batch_time_meter.update(time.time() - end)
    #     if is_odenet:
    #         f_nfe_meter.update(nfe_forward)
    #         b_nfe_meter.update(nfe_backward)
    #     end = time.time()

    #     if iterations % batches_per_epoch == 0:
    #         train_loss /= batches_per_epoch
    #         epoch += 1

    #         with torch.no_grad():
    #             val_acc = accuracy(model, test_loader, args)
    #             logger.info(
    #                 "Epoch {:04d} | Time {:.3f} ({:.3f}) | NFE-F {:.1f} | NFE-B {:.1f} | "
    #                 "Test Acc {:.4f} | Training Loss {:.4f}".format(
    #                     iterations // batches_per_epoch,
    #                     batch_time_meter.val,
    #                     batch_time_meter.avg,
    #                     f_nfe_meter.avg,
    #                     b_nfe_meter.avg,
    #                     val_acc, 
    #                     train_loss))

    #         train_loss = 0.

    #         # Save state to file
    #         if epoch % args.save_interval == 0:
    #             save(epoch, iterations, model, optimizer, args)
