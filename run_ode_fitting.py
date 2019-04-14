import os
import argparse
import time
import numpy as np
import logging
import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim

from util.utils import RunningAverageMeter, makedirs
from os.path import dirname, join

import matplotlib.pyplot as plt

parent_dir = dirname(__file__)
results_dir = join(parent_dir, 'results')
logger = logging.getLogger('neuralODE')

def add_general_arguments(parser):

    parser.add_argument(
        '--method',
        type=str,
        choices=[
            'dopri5',
            'adams'],
        default='dopri5')
    parser.add_argument(
        '--model',
        type=str,
        default='linear',
        metavar='str',
        help='model, choose from spiral, model, spiralNN (default linear)')
    parser.add_argument(
        '--data_size', 
        type=int, 
        default=1000)
    parser.add_argument(
        '--batch_time', 
        type=int, 
        default=10)
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=20)
    parser.add_argument(
        '--niters', 
        type=int, 
        default=2000)
    parser.add_argument(
        '--test_freq', 
        type=int, 
        default=20)
    parser.add_argument(
        '--viz', 
        action='store_true')
    parser.add_argument(
        '--gpu', 
        type=int, 
        default=0)
    parser.add_argument(
        '--adjoint', 
        action='store_true')



def get_batch():
    s = torch.from_numpy(
        np.random.choice(
            np.arange(
                args.data_size -
                args.batch_time,
                dtype=np.int64),
            args.batch_size,
            replace=False))
    batch_y0 = true_y[s]  # (M, D)
    batch_t = t[:args.batch_time]  # (T)
    batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, D)
    return batch_y0, batch_t, batch_y

def visualize(true_y, pred_y, odefunc, itr, imgpath, plot_range):

    if args.viz:

        ax_traj.cla()
        ax_traj.set_title('Trajectories')
        ax_traj.set_xlabel('t')
        ax_traj.set_ylabel('x,y')
        ax_traj.plot(
            t.numpy(), 
            true_y.numpy()[:, 0, 0], 
            t.numpy(), 
            true_y.numpy()[:, 0, 1], 
            'g-')
        ax_traj.plot(
            t.numpy(), 
            pred_y.numpy()[:, 0, 0], 
            '--', 
            t.numpy(), 
            pred_y.numpy()[:, 0, 1], 
            'b--')
        ax_traj.set_xlim(t.min(), t.max())
        ax_traj.set_ylim(plot_range[0], plot_range[1])
        ax_traj.legend()

        ax_phase.cla()
        ax_phase.set_title('Phase Portrait')
        ax_phase.set_xlabel('x')
        ax_phase.set_ylabel('y')
        ax_phase.plot(true_y.numpy()[:, 0, 0], true_y.numpy()[:, 0, 1], 'g-')
        ax_phase.plot(pred_y.numpy()[:, 0, 0], pred_y.numpy()[:, 0, 1], 'b--')
        ax_phase.set_xlim(plot_range[0], plot_range[1])
        ax_phase.set_ylim(plot_range[2], plot_range[3])

        ax_vecfield.cla()
        ax_vecfield.set_title('Learned Vector Field')
        ax_vecfield.set_xlabel('x')
        ax_vecfield.set_ylabel('y')

        y, x = np.mgrid[-2:2:21j, -2:2:21j]
        dydt = odefunc(0, torch.Tensor(
            np.stack([x, y], -1).reshape(21 * 21, 2))).cpu().detach().numpy()
        mag = np.sqrt(dydt[:, 0]**2 + dydt[:, 1]**2).reshape(-1, 1)
        dydt = (dydt / mag)
        dydt = dydt.reshape(21, 21, 2)

        ax_vecfield.streamplot(
            x, y, dydt[:, :, 0], dydt[:, :, 1], color="black")
        ax_vecfield.set_xlim(-2, 2)
        ax_vecfield.set_ylim(-2, 2)

        fig.tight_layout()
        plt.savefig('{}/{:03d}'.format(imgpath,itr))
        plt.draw()
        plt.pause(0.001)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    add_general_arguments(parser)
    args, _ = parser.parse_known_args()

    for arg in vars(args):
        logger.info(f'{arg} {getattr(args, arg)}')

    device = torch.device('cuda:' + str(args.gpu)
                          if torch.cuda.is_available() else 'cpu')

    if args.adjoint:
        from torchdiffeq import odeint_adjoint as odeint
    else:
        from torchdiffeq import odeint

    imgpath = join(results_dir, 'png')


    # initial point of the fitter function
    true_y0 = torch.tensor([[2., 0.]])
    # true_y0 = torch.tensor([[0.6, 0.3]])
    plot_range = [-2,2,-2,2]


    # sample points
    t = torch.linspace(0., 25., args.data_size)
    # true transformer function
    true_A = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]])

    from models import Spiral, Spiral_fit, LinearFunc, LinearFunc_fit, Spiral_NN, Spiral_NN_fit
    model_list = {
        'spiral': {
            Spiral(torch.tensor([[-0.1, 2.0], [-2.0, -0.1]])),
            Spiral_fit(),
        },
        'linear': {
            LinearFunc(),
            LinearFunc_fit(),
        },
        'spiralNN': {
            Spiral_NN(
                Tensor([[-0.1, -0.5], [0.5, -0.1]]), 
                Tensor([[0.2, 1.], [-1, 0.2]]), 
                Tensor([[-1., 0.]])
                ),
            Spiral_NN_fit(2, 16, time_invariant=True),
        },
    }
    
    true_func, func = model_list[args.model]

    with torch.no_grad():
        true_y = odeint(true_func, true_y0, t, method='dopri5')

    if args.viz:
        makedirs(imgpath)

        fig = plt.figure(figsize=(12, 4), facecolor='white')
        ax_traj = fig.add_subplot(131, frameon=False)
        ax_phase = fig.add_subplot(132, frameon=False)
        ax_vecfield = fig.add_subplot(133, frameon=False)
        plt.show(block=False)

    ii = 0

    optimizer = optim.Adam(
        func.parameters(), 
        lr=0.01)
    end = time.time()

    time_meter = RunningAverageMeter(0.97)
    loss_meter = RunningAverageMeter(0.97)

    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        batch_y0, batch_t, batch_y = get_batch()
        pred_y = odeint(func, batch_y0, batch_t)
        loss = torch.mean(torch.abs(pred_y - batch_y))
        loss.backward()
        optimizer.step()

        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())

        if itr % args.test_freq == 0:
            with torch.no_grad():
                pred_y = odeint(func, true_y0, t)
                loss = torch.mean(torch.abs(pred_y - true_y))
                print(
                    'Iter {:04d} | Total Loss {:.6f}'.format(
                        itr, loss.item()))
                visualize(true_y, pred_y, func, ii, imgpath, plot_range)
                ii += 1

        end = time.time()