#!/bin/bash

#fc+MNIST
# python main.py --model-class default --model fc --dataset mnist --experiment singleshot --pruner mag --compression 1 --expid 8 --post-epochs 10 --pre-epochs 200
# python main.py --model-class default --model fc --dataset mnist --experiment singleshot --pruner snip --compression 1 --expid 9 --post-epochs 10
# python main.py --model-class default --model fc --dataset mnist --experiment singleshot --pruner grasp --compression 1 --expid 10 --post-epochs 10


python main.py --model-class lottery --model vgg16 --dataset cifar10 --experiment singleshot --pruner snip --compression 0.05 --expid 4 --post-epochs 100
python main.py --model-class lottery --model vgg16 --dataset cifar10 --experiment singleshot --pruner rand --compression 1 --expid 4 --post-epochs 100
python main.py --model-class lottery --model vgg16 --dataset cifar10 --experiment singleshot --pruner rand --compression 2 --expid 4 --post-epochs 100
python main.py --model-class lottery --model vgg16 --dataset cifar10 --experiment singleshot --pruner mag --compression 2 --expid 3 --post-epochs 100 --pre-epochs 200
python main.py --model-class lottery --model vgg16 --dataset cifar10 --experiment singleshot --pruner synflow --compression 2 --expid 4 --post-epochs 100
python main.py --model-class default --model fc --dataset mnist --experiment singleshot --pruner synflow --compression 1 --expid 6 --post-epochs 10