# neuralODE-282

282 project on neural-ODE

Repo based on [torchdiffeq](https://github.com/rtqichen/torchdiffeq), for the details pase refer to the README of that Repo. 

## Image classification tasks

Run 

```
python run.py
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

## Visualization with Visdom

For now visdom can fetch all the csv files following the particular format and plot them.

Go to the Visdom folder then execute the following commands:
```
visdom -port XXXXX

python visdom_pull_server.py -port XXXXX
```

## Others
Run
```
autopep8 --in-place --aggressive --aggressive *.py
```

before any commit.
