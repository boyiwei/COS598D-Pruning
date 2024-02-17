#!/bin/bash

#fc+MNIST
python main.py --model-class default --model fc --dataset mnist --experiment singleshot --pruner mag --compression 1 --expid 8 --post-epochs 10 --pre-epochs 200
python main.py --model-class default --model fc --dataset mnist --experiment singleshot --pruner snip --compression 1 --expid 9 --post-epochs 10
python main.py --model-class default --model fc --dataset mnist --experiment singleshot --pruner grasp --compression 1 --expid 10 --post-epochs 10

