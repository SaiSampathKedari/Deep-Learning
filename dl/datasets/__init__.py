"""Dataset loaders — one module per dataset (cifar, mnist, ...).

Each module exposes a raw loader and a preprocessing helper; downloads land in
``data/raw/``.
"""
from .cifar import cifar10, preprocess_cifar10

__all__ = ["cifar10", "preprocess_cifar10"]
