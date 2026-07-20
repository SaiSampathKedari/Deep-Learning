# Deep-Learning

Deep learning models implemented from scratch, for learning — losses, gradients,
and training loops written by hand rather than pulled out of a framework.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt   # pinned env, incl. the CUDA 12.1 PyTorch build
pip install -e .                  # makes `import dl` work from anywhere
```

## Layout

```
dl/
├── utils.py          reset_seed, tensor_to_image, visualize_dataset
├── grad_check.py     grad_check_sparse, compute_numeric_gradient, rel_error
└── datasets/         one module per dataset
    └── cifar.py      cifar10, preprocess_cifar10
data/raw/             downloaded datasets (git-ignored)
```

To add a dataset, drop `dl/datasets/<name>.py` with its own loader (downloading
into `data/raw/`) and re-export it from `dl/datasets/__init__.py`. Nothing else moves.

## Usage

```python
import torch
import dl

dl.reset_seed(0)

data = dl.datasets.preprocess_cifar10(bias_trick=True, cuda=True, dtype=torch.float64)
# -> dict with X_train/y_train, X_val/y_val, X_test/y_test

# verify an analytic gradient against a numeric one
dl.grad_check.grad_check_sparse(f, W, grad)   # want relative error < 1e-5
```

CIFAR-10 downloads automatically into `data/raw/` the first time it is used.

> **Run from the repo root.** The dataset path is resolved relative to the current
> working directory, so calling `preprocess_cifar10()` from another directory will
> re-download the data into that directory.

## Dependencies

- `pyproject.toml` declares the **abstract** dependencies — what the package needs.
- `requirements.txt` pins the **concrete** environment — exact versions plus the
  PyTorch CUDA index, since the `+cu121` wheels are not on plain PyPI.
