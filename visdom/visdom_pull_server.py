import visdom
import time
import csv
import os
import pandas as pd
import argparse
import ntpath
import glob
import colorlover as cl
import random

# TODO(yejiayu): Also extend to check models. Probably by path?
parser = argparse.ArgumentParser(description='Visdom server Arguments')
parser.add_argument(
    '--port',
    metavar='p',
    type=int,
    required=True,
    help='Port the visdom server is running on')
parser.add_argument(
    '--root',
    metavar='r',
    type=str,
    default='../results/',
    help='Where to look for CSV files')
parser.add_argument(
    '--interval',
    metavar='s',
    type=int,
    default=10,
    help='Number of seconds to wait before refreshing data')

FLAGS = parser.parse_args()

# Default running on localhost
vis = visdom.Visdom(port=FLAGS.port)


def color_picker(index):
    colors = cl.scales['9']['qual']['Set1']
    if index in color_picker.color_map:
        return colors[color_picker.color_map[index]]
    color_picker.color_map[index] = color_picker.counter
    color_picker.counter += 1
    color_picker.counter %= len(colors)
    return colors[color_picker.color_map[index]]


color_picker.counter = 0
color_picker.color_map = {}


def fetch_all_traces():
    """
        Find all the csv files in a folder and generate a dict of list of traces.
        dict keys: 'train_trace', 'test_trace', 'test_err_trace'
    """
    all_traces = {'train_trace': [], 'test_trace': [], 'test_err_trace': [], 'eigenvalue_trace': []}
    # for all files
    for filename in glob.glob(os.path.join(FLAGS.root, '*.csv')):
        try:
            traces = generate_trace_from_csv(filename)
            for k, v in all_traces.items():
                v.append(traces[k])
        except BaseException as err:
            print('Ignoring error: ', err)

    return all_traces


def generate_trace_from_csv(filename):
    """
        Return a list of [train_trace, test_trace, test_err_trace]
    """
    # filename = '../results/TRCG-QAlexNet.csv'

    df = pd.read_csv(filename, delim_whitespace=True)
    basename = ntpath.basename(filename)
    model_prefix = (os.path.splitext(basename)[0])

    color = color_picker(abs(hash(model_prefix)))
    train_trace = dict(
        x=df['iterations'].tolist(),
        y=df['train_loss'].tolist(),
        mode="markers+lines",
        type='custom',
        marker={'color': color,
                'symbol': 104,
                'size': "1"},
        name=model_prefix)
    test_trace = dict(
        x=df['iterations'].tolist(),
        y=df['test_loss'].tolist(),
        mode="markers+lines",
        type='custom',
        marker={'color': color,
                'symbol': 104,
                'size': "1"},
        name=model_prefix)
    test_err_trace = dict(
        x=df['iterations'].tolist(),
        y=df['test_error'].tolist(),
        mode="markers+lines",
        type='custom',
        marker={'color': color,
                'symbol': 104,
                'size': "1"},
        name=model_prefix)
    eigenvalue_trace = dict(
        x=df['iterations'].tolist(),
        y=df['eigenvalue'].tolist(),
        mode="markers+lines",
        type='custom',
        marker={'color': color,
                'symbol': 104,
                'size': "1"},
        name=model_prefix)
    return {
        'train_trace': train_trace,
        'test_trace': test_trace,
        'test_err_trace': test_err_trace,
        'eigenvalue_trace': eigenvalue_trace
    }


while True:

    all_traces = fetch_all_traces()
    train_layout = dict(
        title="Train loss vs iterations",
        xaxis={'title': 'iterations'},
        yaxis={
            'title': 'train_loss'
        })
    test_layout = dict(
        title="Test loss vs iterations",
        xaxis={'title': 'iterations'},
        yaxis={
            'title': 'test_loss'
        })
    test_err_layout = dict(
        title="Test error vs iterations",
        xaxis={'title': 'iterations'},
        yaxis={
            'title': 'test_error'
        })
    eigenvalue_layout = dict(
        title="Largest eigenvalue vs iterations",
        xaxis={'title': 'iterations'},
        yaxis={
            'title': 'Largest_eigenvalue'
        })

    vis._send({
        'data': all_traces['train_trace'],
        'layout': train_layout,
        'win': 'win1'
    })
    vis._send({
        'data': all_traces['test_trace'],
        'layout': test_layout,
        'win': 'win2'
    })
    vis._send({
        'data': all_traces['test_err_trace'],
        'layout': test_err_layout,
        'win': 'win3'
    })
    vis._send({
        'data': all_traces['eigenvalue_trace'],
        'layout': eigenvalue_layout,
        'win': 'win4'
    })
    time.sleep(FLAGS.interval)
