# neuralODE - Berkeley CS 282 project

report: [An empirical study of neural ordinal differential equations](https://linjianma.github.io/pdf/282_project_report_ode.pdf)

The ODE kernel is based on [torchdiffeq](https://github.com/rtqichen/torchdiffeq), for the details please refer to the README of that Repo. 

## Image classification tasks

Run 

```
python run_image_classification.py
```
To compare NeuralODE with ResNet on MNIST / CIFAR10. 

Some running examples:
```
python run_image_classification.py --epochs 200 --dataset cifar10 --batch-size 512 --double "" --network odenet --gpu 1 --lr 0.1 --momentum 0.95 --adjoint 1 --method midpoint
```

## Function fitting tasks

Some running examples:
```
python run_ode_fitting.py --model spiralNN --viz --adjoint
```

## Time series tasks

Model and code are based on the paper: [Deep Multi-Output Forecasting](https://arxiv.org/pdf/1806.05357.pdf)

Running examples:
```
python multi-output-glucose-forecasting/run.py
```

## Visualization with Visdom

For now visdom can fetch all the csv files following the particular format and plot them.

Go to the Visdom folder then execute the following commands:
```
visdom -port XXXXX

python visdom_pull_server.py -port XXXXX
```
