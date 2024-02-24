# Network Pruning
### Assignment 1 for COS598D: System and Machine Learning

In this assignment, you are required to evaluate three advanced neural network pruning methods, including SNIP [1], GraSP [2] and SynFlow [3], and compare with two baseline pruning methods, including random pruning and magnitude-based pruning. In `example/singleshot.py`, we provide an example to do singleshot global pruning without iterative training. In `example/multishot.py`, we provide an example to do multi-shot iterative training. This assignment focuses on the pruning protocol in `example/singleshot.py`. Your are going to explore various pruning methods on different hyperparameters and network architectures.

***References***

[1] Lee, N., Ajanthan, T. and Torr, P.H., 2018. Snip: Single-shot network pruning based on connection sensitivity. arXiv preprint arXiv:1810.02340.

[2] Wang, C., Zhang, G. and Grosse, R., 2020. Picking winning tickets before training by preserving gradient flow. arXiv preprint arXiv:2002.07376.

[3] Tanaka, H., Kunin, D., Yamins, D.L. and Ganguli, S., 2020. Pruning neural networks without any data by iteratively conserving synaptic flow. arXiv preprint arXiv:2006.05467.

### Additional reading materials:

A recent paper [4] assessed [1-3].

[4] Frankle, J., Dziugaite, G.K., Roy, D.M. and Carbin, M., 2020. Pruning Neural Networks at Initialization: Why are We Missing the Mark?. arXiv preprint arXiv:2009.08576.

## Getting Started
First clone this repo, then install all dependencies
```
pip install -r requirements.txt
```

## How to Run 
Run `python main.py --help` for a complete description of flags and hyperparameters. You can also go to `main.py` to check all the parameters. 

Example: Initialize a VGG16, prune with SynFlow and train it to the sparsity of 10^-0.5 . We have sparsity = 10**(-float(args.compression)).
```
python main.py --model-class lottery --model vgg16 --dataset cifar10 --experiment singleshot --pruner synflow --compression 0.5
```

To save the experiment, please add `--expid {NAME}`. `--compression-list` and `--pruner-list` are not available for runing singleshot experiment. You can modify the souce code following `example/multishot.py` to run a list of parameters. `--prune-epochs` is also not available as it does not affect your pruning in singleshot setting. 

For magnitude-based pruning, please set `--pre-epochs 200`. You can reduce the epochs for pretrain to save some time. The other methods do pruning before training, thus they can use the default setting `--pre-epochs 0`.

Please use the default batch size, learning rate, optimizer in the following experiment. Please use the default training and testing spliting. Please monitor training loss and testing loss, and set suitable training epochs. You may try `--post-epoch 100` for Cifar10 and `--post-epoch 10` for MNIST.

If you are using Google Colab, to accommodate the limited resources on Google Colab, you could use `--pre-epochs 10` for magnitude pruning and use `--post-epoch 10` for cifar10 for experiments on Colab. And state the epoch numbers you set in your report.

## You Tasks

### 1. Hyper-parameter tuning

#### Testing on different archietectures. Please fill the results table:
*Test accuracy (top 1)* of pruned models on CIFAR10 and MNIST (sparsity = 10%). `--compression 1` means sparsity = 10^-1.
```
python main.py --model-class lottery --model vgg16 --dataset cifar10 --experiment singleshot --pruner synflow --compression 1 --expid 1
```
```
python main.py --model-class default --model fc --dataset mnist --experiment singleshot --pruner synflow --compression 1 --expid 2
```
***Testing accuracy (top 1)***

|   Data  |   Arch |   Rand |  Mag |  SNIP |  GraSP | SynFlow       |   
|----------------|----------------|-------------|-------------|-------------|---------------|----------------|
|Cifar10 | VGG16 |  10.0  |   89.53   |   86.64     | 28.4    |     87.94    |
|MNIST| FC |   94.49 |   97.72   |   95.54     |   94.69   |  11.35       |


#### Tuning compression ratio. Please fill the results table:
Prune models on CIFAR10 with VGG16, please replace {} with sparsity 10^-a for a \in {0.05,0.1,0.2,0.5,1,2}. Feel free to try other sparsity values.

```
python main.py --model-class lottery --model vgg16 --dataset cifar10 --experiment singleshot --pruner synflow  --compression {}
```
***Testing accuracy (top 1)***

|   Compression |   Rand |  Mag |  SNIP |  GraSP | SynFlow       |   
|----------------|-------------|-------------|-------------|---------------|----------------|
| 0.05|  88.24  | 88.8     |        |   50.37  |   88.51      |
| 0.1| 87.5   |    89.72  |    87.84    |  51.79   |   88.41      |
| 0.2|  87.8  |   89.61   |   88.6     |  42.45    |  87.97       |
| 0.5| 87.13   | 89.76    |   87.1     |  20.46    |   88.46      |
| 1|  10.0  |  89.53    |   86.64     |  28.4    |    87.94     |
| 2|  10.0  |   19.3   |    82.07    |  59.42    |   10.0      |

***Testing time (inference on testing dataset)***

|   Compression |   Rand |  Mag |  SNIP |  GraSP | SynFlow       |   
|----------------|-------------|-------------|-------------|---------------|----------------|
| 0.05|  1.45950  |  1.49481    |       |  1.47357   |     1.47234    |
| 0.1|  1.48594  |   1.49217   |    1.49227    |  1.49889   |   1.47147      |
| 0.2|  1.46599  |   1.51180   |   1.50174     |  1.48238    |   1.48203      |
| 0.5|  1.47389  |   1.48229   |   1.48837      |  1.54439    |   1.48598      |
| 1|  2.07546  |   1.51999   |    1.50930    |   1.52564   |   1.52116      |
| 2|  1.47549  |   1.54343  |    1.47972    |   1.50731   |   1.50812      |

To track the runing time, you can use `timeit`. `pip intall timeit` if it has not been installed.
```
import timeit

start = timeit.default_timer()

#The module that you try to calculate the running time

stop = timeit.default_timer()

print('Time: ', stop - start)
```


***FLOP***

|   Compression |   Rand |  Mag |  SNIP |  GraSP | SynFlow       |   
|----------------|-------------|-------------|-------------|---------------|----------------|
| 0.05|  0.89151 |   0.94856   |        |  0.82018   |    0.94876     |
| 0.1|   0.79449  | 0.90159     |  0.92676      | 0.72949    |  0.90258       |
| 0.2|  0.63115  |  0.81854    |  0.78122      |  0.57101    |  0.82169       |
| 0.5|  0.31626  |   0.57860  |  0.46206      |   0.37802   |   0.64233      |
| 1| 0.10074   |   0.22389   |   0.19792     |   0.17507   |    0.45713     |
| 2|  0.01085  |  0.03134    |    0.04101    | 0.05858     |   0.18448      |

For better visualization, you are encouraged to transfer the above three tables into curves and present them as three figrues.
### 2. The compression ratio of each layer
Report the sparsity of each layer and draw the weight histograms of each layer using pruner Rand |  Mag |  SNIP |  GraSP | SynFlow with the following settings
`model = vgg16`, `dataset=cifar10`, `compression = 0.5`

Weight histogram is a figure showing the distribution of weight values. Its x axis is the value of each weight, y axis is the count of that value in the layer. Since the weights are floating points, you need to partite the weight values into multiple intervals and get the numbers of weights which fall into each interval. The weight histograms of all layers of one pruning method can be plotted in one figure (one histogram for each layer).

This is an example of weight histograms for NN
https://stackoverflow.com/questions/42315202/understanding-tensorboard-weight-histograms

***Bonus (optional)***

Report the FLOP of each layer using pruner Rand |  Mag |  SNIP |  GraSP | SynFlow with the following settings
`model = vgg16`, `dataset=cifar10`, `compression= 0.5`.
### 3. Explain your results and submit a short report.
Please describe the settings of your experiments. Please include the required results (described in Task 1 and 2). Please add captions to describe your figures and tables. It would be best to write brief discussions on your results, such as the patterns (what and why), conclusions, and any observations you want to discuss.  
