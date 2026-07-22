# Deep-Learning

Deep learning models implemented from scratch in PyTorch.

## Contents

| Component | Details |
|---|---|
| Linear classifiers | Multiclass SVM and Softmax, naive and vectorized, with analytic gradients |
| Two-layer network | `affine -> ReLU -> affine` with manual backprop |
| Layers | `Linear`, `ReLU`, `Dropout`, `Linear_ReLU`, `SoftmaxCrossEntropy` |
| Fully connected network | Arbitrary depth, optional dropout, L2 regularization |
| Optimizers | SGD, SGD with momentum, RMSProp, Adam |
| Solver | Training loop with checkpointing and learning rate decay |
| Gradient checking | Sparse spot checks and full numeric gradients |

## Results

CIFAR-10, CPU, seed 0. Reproducible from `notebooks/`.

| Model | Val | Test |
|---|---|---|
| Linear SVM | 0.3849 | 0.3818 |
| Softmax | 0.4000 | 0.4034 |
| Two-layer net | 0.4503 | 0.4686 |

## Layout

```
dl/
├── nn/          layers, file names follow torch/nn/modules
├── optim.py     sgd, sgd_momentum, rmsprop, adam
├── solver.py    training loop
├── models/      one module per architecture
├── datasets/    one module per dataset
├── grad_check.py
└── utils.py

notebooks/       experiments, the .py files are the source of truth
data/raw/        downloaded datasets (git-ignored)
checkpoints/     saved weights (git-ignored)
```

## Setup

```bash
pip install -r requirements.txt
pip install -e .
```

CIFAR-10 downloads into `data/raw/` on first use. Set `DL_DATA_ROOT` to change
the location.

## Reference

Follows the assignments from [EECS 498-007 / 598-005: Deep Learning for Computer
Vision](https://web.eecs.umich.edu/~justincj/teaching/eecs498/), University of
Michigan.

MIT licensed.
