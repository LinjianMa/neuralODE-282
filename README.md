# neuralODE-282

282 project on neural-ODE

Repo based on [torchdiffeq](https://github.com/rtqichen/torchdiffeq), for the details pase refer to the README of that Repo. 

## Visualization with Visdom

For now visdom can fetch all the csv files following the particular format and plot them.

Go to the Visdom folder then execute the following commands:
```
visdom -port XXXXX

python visdom_pull_server.py -port XXXXX
```